import tensorflow_datasets as tfds
import numpy as np
import jax
from jax import jit, vmap, pmap
import jax.numpy as jnp
import util as u


def shard_dataset(training):
    # choose split that divides evenly across 4 hosts
    h = jax.host_id()
    if training:  # train[:80%]
        split = f"train[{h*5400}:{(h+1)*5400}]"
    else:  # validation; # train[80%:90%]
        split = f"train[{21600+(h*675)}:{21600+((h+1)*675)}]"
    print("for host", jax.host_id(), "split is", split)

    # load images and labels as single batch
    dataset = tfds.load('eurosat/rgb', split=split, as_supervised=True)
    for tfe_imgs, tfe_labels in dataset.batch(100_000):
        break

    # clip size to ensure can be reshaped to leading 8
    imgs, labels = np.array(tfe_imgs), np.array(tfe_labels)
    del tfe_imgs  # transistent OOM ??
    del tfe_labels
    n = (len(imgs) // 8) * 8
    imgs, labels = imgs[:n], labels[:n]
    # resize to leading dim of 8
    imgs = imgs.reshape(8, n // 8, 64, 64, 3)
    labels = labels.reshape(8, n // 8)
    # return sharded and with images converted

    def convert(a):
        return jnp.array(a / 255, dtype=jnp.float32)
    return pmap(convert)(imgs), u.shard(labels)


def rot90(m, k=1):
    # jax.numpy.rot90 uses conditionals on k which means we
    # can't vmap over them. but if we swap them to jnp.wheres
    # we can!
    # https://jax.readthedocs.io/en/latest/_modules/jax/_src/numpy/lax_numpy.html#rot90
    ax1, ax2 = 0, 1
    k = k % 4

    def not_0():
        return jnp.where(k == 2,
                         jnp.flip(jnp.flip(m, ax1), ax2),
                         not_0_or_2())

    def not_0_or_2():
        perm = list(range(m.ndim))
        perm[ax1], perm[ax2] = perm[ax2], perm[ax1]
        return jnp.where(k == 1,
                         jnp.transpose(jnp.flip(m, ax2), perm),
                         jnp.flip(jnp.transpose(m, perm), ax2))

    return jnp.where(k == 0, m, not_0())


def random_flip(m, k):
    return jnp.where(k == 0, m, jnp.fliplr(m))


def augment(img, rot, flip):
    img = rot90(img, rot)
    img = random_flip(img, flip)
    return img


v_augment = vmap(augment, in_axes=(None, 0, 0))


def all_combos_augment(img):  # (H,W,C) -> (8,H,W,C)
    rots = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    flips = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return v_augment(img, rots, flips)


@pmap
def v_all_combos_augment(imgs):
    imgs = vmap(all_combos_augment)(imgs)  # (B,H,W,C) -> (B,8,H,W,C)
    return imgs.reshape(-1, 64, 64, 3)     # (8B, H, W, C)

# all_combo_augment = jit(vmap(all_combo_augment))  # (B,H,W,C) -> (B,8,H,W,C)
