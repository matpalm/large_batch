
try:
    import jax_pod_setup
except ModuleNotFoundError:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import data as d
import util as u

train_imgs, train_labels = d.shard_dataset(training=True)
validate_imgs, validate_labels = d.shard_dataset(training=False)

print(u.shapes_of(train_imgs), u.shapes_of(train_labels),
      u.shapes_of(validate_imgs), u.shapes_of(validate_labels))
