from multiprocessing.resource_sharer import stop
from posixpath import split
from tkinter import Y
from tracemalloc import start
import jax
import jax.numpy as jnp
from jax import jit

# a = jnp.load('/home/jungang/jax_privacy/jax_privacy/src/training/change_num_list.npy')
# num = jax.lax.dynamic_slice(a, (2,), (12,))
# print(num)
a = jnp.arange(12).reshape(3,4)
print(a)
q = jnp.array([10,20,50,80])
a_3 = jnp.percentile(a,q, axis=1)
a_4  = jnp.percentile(a,q, axis=0)
# a_5 = jnp.percentile(a,q, axis=1, keepdims=True)

print(a_3.shape)
print(jnp.expand_dims(jnp.diag(a_4),0).repeat(3, axis=0))

# print(jnp.diag(a_3).repeat(3).reshape(2,3))
# print(a_4)
# print(jnp.diag(a_5))





# pruning_ratio = 0.1
# N = 3*3*256*256
# n = int(N * pruning_ratio)

# x = jnp.arange(start=1, stop=n+1)
# log_x = jnp.log(x)

# y = jnp.arange(start= N - 2 * n + 1, stop=N - n + 1)
# log_y = jnp.log(y)
# res_1 = jnp.sum(log_y) - jnp.sum(log_x)
# print(log_y)


# M = 256
# part_num = N // M
# print(part_num)
# m = int(M * pruning_ratio)
# xm = jnp.arange(start=1, stop=m+1)
# log_xm = jnp.log(xm)
# ym = jnp.arange(start= M - 2 * m + 1, stop= M - m + 1)
# log_ym = jnp.log(ym)
# res_2 = jnp.sum(log_xm) - jnp.sum(log_ym)
# print(res_1)
# print(res_2)
# print(res_1 + res_2*part_num)



# y = jnp.arange(start=256, stop=x.size, step=256)
# print(y)
# print(x.size//256)
# split_x = jnp.array_split(x, x.size // 256 + 1)
# print(split_x[0].shape)
# print(jnp.concatenate([x,x]))
# # print(split_x[1].shape)
# # print(split_x[2].shape)
# # print(split_x[3].shape)
# print(split_x)

# key = jax.random.PRNGKey(111)
# key, subkey = jax.random.split(key)

# @jax.jit
# def getRes(x):
#     def get_mask(x):
#         # print('x', x)
#         random_grad = jax.random.normal(key, jnp.shape(x))
#         quantile=jnp.percentile(jnp.abs(random_grad), 100-50)
#         return jnp.where(jnp.abs(random_grad)> quantile, jnp.ones_like(x), jnp.zeros_like(x))

#     # y = jnp.arange(start=255, stop=1000, step=256)
#     split_x = jnp.array_split(x, x.size // 256 + 1)
#     res = jax.tree_util.tree_map(lambda x: x * get_mask(x), split_x)
#     # print(jnp.concatenate(res))
#     return res
# print(getRes(x))
# print(jnp.where(getRes(x)==0, 1, 0))

# import jax
# import jax.numpy as jnp
# from jax import grad, jit, vmap
# from jax import random
# import functools
# import chex
# from typing import Callable, Iterator, Mapping, Sequence, Tuple
# import numpy as np


# # XLA_FLAGS=1

# def tanh(x):  # Define a function
#   y = jnp.exp(-2.0 * x)
#   return (1.0 - y) / (1.0 + y)

# grad_tanh = grad(tanh)  # Obtain its gradient function
# print(grad_tanh(1.0))   # Evaluate it at x = 1.0

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# size = 3000
# x = random.normal(jax.random.PRNGKey(42), (size, size), dtype=jnp.float32)
# %timeit jnp.dot(x, x.T).block_until_ready()  # runs on the GPU

# Params = chex.ArrayTree
# GradParams = Params
# key = random.PRNGKey(111)
# key, subkey = random.split(key)
# # Batch = Mapping[str, np.ndarray]
# Predicate = Callable[[str, str, jnp.ndarray], bool]
# PredicateMap = Mapping[Predicate, jnp.ndarray]
# ModuleSparsity = Sequence[Tuple[Predicate, jnp.ndarray]]
# # A list as a pytree
# example_1 = {'weight': [1., 2., 3., 4.], 'bias': [2., 5., 6., 8.]}
# example_2 = {'weight': [2., 2., 3., 4.], 'bias': [9., 5., 6., 8.]}
# # As in normal Python code, a list that represents pytree
# # can contain obejcts of any type
# # example_2 = random.normal(key=key, shape=(12, 1))

# # Similarly we can define pytree using a tuple as well
# example_3 = random.normal(key=key, shape=(8, 1))

# # We can define the same pytree using a dict as well
# example_4 = random.normal(key=key, shape=(8, 1))

# # Let's check the number of leaves and the corresponding values in the above pytrees
# example_pytrees = [example_1, example_2]

# example_pytrees_2 = [example_3, example_4]

# # print(example_pytrees)
# def get_mask(x):
#   print(x)
#   return  x

# def _get_mask(x):
#     quantile=jnp.percentile(jnp.abs(x),1)
#     mask = jnp.where(jnp.abs(x) > quantile, jnp.zeros_like(x), jnp.ones_like(x))
#     return  mask

# print(jax.tree_map(lambda x: get_mask(x), example_pytrees))
# print(jax.tree_map(lambda x: _get_mask(x), example_pytrees))
# print(jax.tree_map(lambda x: _get_mask(x), example_pytrees_2))
# print(jax.vmap(get_mask, in_axes=({"weights":0, 'bias':0})(example_pytrees)))
