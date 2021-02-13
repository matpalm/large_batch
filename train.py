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


class Options(object):
    max_conv_size = 8  # 256
    dense_kernel_size = 8  # 96
    seed = 123
    learning_rate = 0.001
    weight_decay = 0.001
    force_small_data = True


opts = Options()

train_imgs, train_labels = d.shard_dataset(training=True,
                                           force_small_data=opts.force_small_data)
validate_imgs, validate_labels = d.shard_dataset(training=False,
                                                 force_small_data=opts.force_small_data)

logging.info("loaded %s %s %s %s",
             u.shapes_of(train_imgs), u.shapes_of(train_labels),
             u.shapes_of(validate_imgs), u.shapes_of(validate_labels))

augmented_train_imgs = d.v_all_combos_augment(train_imgs)  # # (8, 8B, H, W, C)
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

host_rng = jax.random.PRNGKey(opts.seed ^ jax.host_id())  # unused?
pod_rng = jax.random.PRNGKey(opts.seed - 1)

representative_input = jnp.zeros((1, 64, 64, 3))
pod_rng, init_key = jax.random.split(pod_rng)
params = model.init(init_key, representative_input)

# construct optimiser

opt = optax.lamb(learning_rate=opts.learning_rate,
                 weight_decay=opts.weight_decay)

opt_state = opt.init(params)

# replicate both across devices

params = u.replicate(params)
opt_state = u.replicate(opt_state)

logging.info("replicated params %s", u.shapes_of(params))
logging.info("replicated opt_state %s", u.shapes_of(opt_state))

# define training loops and some validation functions


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def mean_cross_entropy(params, x, y_true):
    logits = model.apply(params, x)
    return jnp.mean(softmax_cross_entropy(logits, y_true))


def update(params, opt_state, x, y_true):
    # calc grads; summed across devices
    loss, grads = value_and_grad(mean_cross_entropy)(params, x, y_true)
    grads = tree_map(lambda v: psum(v, 'device'), grads)
    # apply update
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    # return new states & mean loss
    return params, opt_state, loss.mean()


p_update = pmap(update, in_axes=(0, 0, 0, 0), axis_name='device')


def predictions(params, x):
    logits = model.apply(params, x)
    return jnp.argmax(logits, axis=-1)


p_predictions = pmap(predictions, in_axes=(0, 0))


def accuracy(params, x, y_true):
    y_pred = p_predictions(params, x)
    num_correct = jnp.sum(jnp.equal(y_pred, y_true))
    num_total = x.shape[0] * x.shape[1]  # recall; x is sharded!
    return float(num_correct / num_total)

# run simple training loop


for outer_idx in range(20):
    for inner_idx in range(30):
        params, opt_state, loss = p_update(params, opt_state,
                                           augmented_train_imgs,
                                           augmented_train_labels)
        logging.info("%s %s", outer_idx, inner_idx)
    train_accuracy = accuracy(params, train_imgs, train_labels)
    validation_accuracy = accuracy(params, validate_imgs, validate_labels)
    logging.info(f"last train mean loss {float(loss.mean()):0.3f}"
                 f" train accuracy {train_accuracy:0.2f}"
                 f" validation accuracy {validation_accuracy:0.2f}")
