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
from jax import pmap
import jax
import jax.numpy as jnp
import model
import haiku as hk
from functools import partial

train_imgs, train_labels = d.shard_dataset(training=True)
validate_imgs, validate_labels = d.shard_dataset(training=False)

logging.info("loaded %s %s %s %s",
             u.shapes_of(train_imgs), u.shapes_of(train_labels),
             u.shapes_of(validate_imgs), u.shapes_of(validate_labels))

augmented_train_imgs = d.v_all_combos_augment(train_imgs)  # # (8, 8B, H, W, C)
augmented_train_labels = pmap(lambda v: jnp.repeat(v, 8))(train_labels)

logging.info("augmented %s %s",
             u.shapes_of(augmented_train_imgs),
             u.shapes_of(augmented_train_labels))


class Options(object):
    max_conv_size = 8  # 256
    dense_kernel_size = 8  # 96
    seed = 123
    learning_rate = 0.001
    weight_decay = 0.001


opts = Options()


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

logging.info("params %s", u.shapes_of(params))
