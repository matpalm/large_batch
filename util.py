import jax.numpy as jnp
from jax import pmap
from jax.tree_util import tree_map

def shard(x):
    # pmap x across first axis
    return pmap(lambda v: v)(x)

def replicate(x, replicas=8):
    # replicate leafs of x and then shard
    replicated = tree_map(lambda v: jnp.stack([v] * replicas), x)
    return shard(replicated)

def shapes_of(pytree):
    # rebuild a pytree swapping actual params for just shape and type
    return tree_map(lambda v: (v.shape, type(v), v.dtype), pytree)