try:
    import jax_pod_setup
except ModuleNotFoundError:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import logging
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO)

import data as d
import util as u
import jax
from jax import pmap, value_and_grad
import jax.numpy as jnp
import model
import haiku as hk
from functools import partial
import optax
from jax.tree_util import tree_map
from jax.lax import psum
import argparse
import wandb
import time

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--group', type=str,
                    help='w&b init group', default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max-conv-size', type=int, default=256)
parser.add_argument('--dense-kernel-size', type=int, default=128)
parser.add_argument('--optimiser', type=str, default='lamb',
                    help='optimiser to use. {adam,lamb,sgd}')
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay. only applicable to lamb.')
parser.add_argument('--momentum', type=float, default=0.0,
                    help='momentum. only applicable to sgd.')
parser.add_argument('--num-outer-steps', type=int, default=50,
                    help='number of times to run inner step')
parser.add_argument('--num-inner-steps', type=int, default=20,
                    help='number of steps to do between each validation check')
parser.add_argument('--force-small-data', action='store_true',
                    help='if set only use 10 instances for training and'
                         ' validation')

opts = parser.parse_args()
logging.info("opts %s", opts)

if opts.optimiser not in ['adam', 'lamb', 'sgd']:
    raise Exception("invalid --optimiser")

run = u.DTS()
logging.info("starting run %s", run)

# only run wandb stuff if it's configured, and only on primary host
wandb_enabled = (opts.group is not None) and u.primary_host()

if wandb_enabled:
    wandb.init(project='large_batch', group=opts.group, name=run,
               reinit=True)
    # save group again explicitly to work around sync bug that drops
    # group when 'wandb off'
    wandb.config.group = opts.group
    wandb.config.seed = opts.seed
    wandb.config.max_conv_size = opts.max_conv_size
    wandb.config.dense_kernel_size = opts.dense_kernel_size
    wandb.config.optimiser = opts.optimiser
    wandb.config.learning_rate = opts.learning_rate
    wandb.config.weight_decay = opts.weight_decay
    wandb.config.momentum = opts.momentum
    wandb.config.num_outer_steps = opts.num_outer_steps
    wandb.config.num_inner_steps = opts.num_inner_steps
else:
    logging.info("not using wandb and/or not primary host")

# load host's worth of training and validation data

train_imgs, train_labels = d.shard_dataset(
    training=True, force_small_data=opts.force_small_data)
validate_imgs, validate_labels = d.shard_dataset(
    training=False, force_small_data=opts.force_small_data)

logging.info("loaded %s %s %s %s",
             u.shapes_of(train_imgs), u.shapes_of(train_labels),
             u.shapes_of(validate_imgs), u.shapes_of(validate_labels))

# augment training data with x8 values

# (8, B, H, W, C) -> (8, 8B, H, W, C)
augmented_train_imgs = d.batched_all_combos_augment(train_imgs)
augmented_train_labels = pmap(lambda v: jnp.repeat(v, 8))(train_labels)

logging.info("augmented %s %s",
             u.shapes_of(augmented_train_imgs),
             u.shapes_of(augmented_train_labels))


# construct model

def build_model(opts):
    m = partial(model.haiku_model,
                max_conv_size=opts.max_conv_size,
                dense_kernel_size=opts.dense_kernel_size)
    return hk.without_apply_rng(hk.transform(m))


model = build_model(opts)

# host_rng = jax.random.PRNGKey(opts.seed ^ jax.host_id())  # unused
pod_rng = jax.random.PRNGKey(opts.seed - 1)

representative_input = jnp.zeros((1, 64, 64, 3))
pod_rng, init_key = jax.random.split(pod_rng)
params = model.init(init_key, representative_input)

# construct optimiser
if opts.optimiser == 'adam':
    opt = optax.adam(learning_rate=opts.learning_rate)
elif opts.optimiser == 'lamb':
    opt = optax.lamb(learning_rate=opts.learning_rate,
                     weight_decay=opts.weight_decay)
else:  # sgd
    opt = optax.sgd(learning_rate=opts.learning_rate,
                    momentum=opts.momentum)

opt_state = opt.init(params)

# replicate both across devices

params = u.replicate(params)
opt_state = u.replicate(opt_state)

# define training loops and some validation functions


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def mean_cross_entropy(params, x, y_true):
    logits = model.apply(params, x)
    return jnp.mean(softmax_cross_entropy(logits, y_true))


@partial(pmap, in_axes=(0, 0, 0, 0), axis_name='device')
def update(params, opt_state, x, y_true):
    # calc grads; summed across devices
    loss, grads = value_and_grad(mean_cross_entropy)(params, x, y_true)
    grads = tree_map(lambda v: psum(v, 'device'), grads)
    # apply update
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    # return new states & mean loss
    return params, opt_state, loss.mean()


@partial(pmap, in_axes=(0, 0))
def predictions(params, x):
    logits = model.apply(params, x)
    return jnp.argmax(logits, axis=-1)


def accuracy(params, x, y_true):
    y_pred = predictions(params, x)
    num_correct = jnp.sum(jnp.equal(y_pred, y_true))
    num_total = x.shape[0] * x.shape[1]  # recall; x is sharded!
    return float(num_correct / num_total)

# run simple training loop


best_validation_accuracy = 0.0
best_validation_idx = None

for outer_idx in range(opts.num_outer_steps):

    inner_step_start = time.time()
    for inner_idx in range(opts.num_inner_steps):
        params, opt_state, loss = update(params, opt_state,
                                         augmented_train_imgs,
                                         augmented_train_labels)
    step_duration = time.time() - inner_step_start

    train_accuracy = accuracy(params, train_imgs, train_labels)
    validation_accuracy = accuracy(params, validate_imgs, validate_labels)
    last_mean_loss = float(loss.mean())

    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        best_validation_idx = outer_idx

    logging.info(f"{outer_idx}: inner_step_duration {step_duration:0.5f}"
                 f" last train mean loss {last_mean_loss:0.3f}"
                 f" train accuracy {train_accuracy:0.2f}"
                 f" validation accuracy {validation_accuracy:0.2f}")

    if wandb_enabled:
        wandb.log({'last_mean_loss': last_mean_loss}, step=outer_idx)
        wandb.log({'train_accuracy': train_accuracy}, step=outer_idx)
        wandb.log({'validation_accuracy': validation_accuracy}, step=outer_idx)

if wandb_enabled:
    wandb.log({'best_validation_accuracy': best_validation_accuracy,
               'best_validation_idx': best_validation_idx},
              step=opts.num_outer_steps)
    wandb.join()

logging.info("run %s best_validation_accuracy %f best_validation_idx %d",
             run, best_validation_accuracy, best_validation_idx)
