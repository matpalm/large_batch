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
import jax.numpy as jnp

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
