import jax.numpy as jnp
import haiku as hk
import jax


def global_spatial_mean_pooling(x):
    return jnp.mean(x, axis=(1, 2))


def haiku_model(x, dense_kernel_size=64, max_conv_size=256, num_classes=10):
    layers = []
    for i, c in enumerate([32, 64, 128, 256]):
        c = min(c, max_conv_size)
        layers.append(hk.Conv2D(output_channels=c, kernel_shape=3, stride=2,
                                name="conv%d_%d" % (i, c)))
        layers.append(jax.nn.gelu)
    layers += [global_spatial_mean_pooling,
               hk.Linear(dense_kernel_size,
                         name="dense_%d" % dense_kernel_size),
               jax.nn.gelu,
               hk.Linear(num_classes, name='logits')]
    return hk.Sequential(layers)(x)
