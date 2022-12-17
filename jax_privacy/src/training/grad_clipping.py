  # coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Computing gradients that are clipped per sample."""

from functools import total_ordering
from re import T
from typing import Callable, Tuple
import numpy as np

# from numpy import mask_indices

import chex
from haiku import vmap
import jax
import jax.numpy as jnp
from jax_privacy.src.training import grad_clipping_utils
import optax

import jax.tree_util
import tree
#from jax_privacy.src.training.mallowsmodel import mallows_model

Aux = chex.ArrayTree
Inputs = chex.ArrayTree
ModelState = chex.ArrayTree
Loss = chex.Array
Params = chex.ArrayTree
GradParams = Params
GradNorms = chex.Array


ClippingFn = Callable[[GradParams], Tuple[GradParams, Aux]]
PruningFn = Callable[[GradParams], Tuple[GradParams, Aux]]
GradFn = Callable[[Params, Inputs, ModelState, chex.PRNGKey],
                  Tuple[Tuple[Loss, Aux], Tuple[GradParams, Aux]]]
LossFn = Callable[[Params, Inputs, ModelState, chex.PRNGKey], Tuple[Loss, Aux]]


def safe_div(
    numerator: chex.Array,
    denominator: chex.Array,
    eps: chex.Numeric = 1e-10,
) -> ClippingFn:
  """Numerically safe division."""
  return numerator / (denominator + eps)


def _placeholder_like(*args):
  return jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), args)


def global_clipping(
    clipping_norm: chex.Array,
    rescale_to_unit_norm: bool = False,
    eps: chex.Numeric = 1e-10,
) -> ClippingFn:
  """Create a function that clips its input tree to have a maximum L2 norm.

  The L2 norm is computed across leaves of the tree. If the input tree has an L2
  norm that is less or equal to `clipping_norm`, it is left untouched by the
  clipping operation. Otherwise it is scaled down by a positive factor so that
  its new L2 norm is exactly `clipping_norm`.

  Note that the clipping function will return NaN entries if the numerical
  constant `eps` is not small enough. This is to loudly detect loss of
  numerical precision that could lead to invalid results.

  Args:
    clipping_norm: maximum L2 norm to which the input tree should be clipped.
    rescale_to_unit_norm: whether the tree should be rescaled to have an L2
      norm of one once it got clipped.
    eps: small numerical constant for numerical stability.
  Returns:
    Function that clips its input tree to have a maximum L2 norm of
    `clipping_norm`.
  """

  def coeff_fn(tree_norm):
    if rescale_to_unit_norm:
      # coeff = min(1, clipping_norm / tree_norm) / clipping_norm
      return jnp.minimum(
          safe_div(1.0, clipping_norm, eps),
          safe_div(1.0, tree_norm, eps)
      )
    else:
      # coeff = min(1, clipping_norm / tree_norm)S
      return jnp.minimum(1.0, safe_div(clipping_norm, tree_norm, eps))

  def clipping_fn(grad: GradParams) -> Tuple[GradParams, Aux]:
    grad_norm = optax.global_norm(grad)
    # print("************************", clipping_norm)
    # If the value of `eps` is invalid because it is too large compared to
    # `clipping_norm`, propagate NaNs to show that the computation is invalid.
    # Note: this has the side effect of always back-propagating NaNs if we
    # differentiate through this function, but this function is not meant to
    # be differentiated, since it post-processes gradients in order to
    # privatize them.
    coeff = jnp.where(clipping_norm > eps, coeff_fn(grad_norm), jnp.nan)
    #quantile = jnp.percentile(coeff,10)
    
    return jax.tree_util.tree_map(lambda x: x * coeff, grad), grad_norm, grad

  return clipping_fn

def pruning(  ##
    pruning_amount: chex.Array,
) -> PruningFn:
  def pruning_fn(grad: GradParams) -> Tuple[GradParams, Aux]:
    tree_value, tree_def=jax.tree_flatten(grad)
    tree_value=map(leaves_pruning,tree_value)
    #tree_value=jax.vmap(lambda x: leaves_pruning(x), tree_value)
    #vmap
    tree_output=jax.tree_unflatten(tree_def,tree_value)
    return tree_output

  def leaves_pruning(leaves):
    quantile=jnp.percentile(jnp.abs(jnp.array(leaves)), 100-pruning_amount)
    return jax.tree_util.tree_map(lambda x: x * get_mask(x, quantile), leaves)

  # def leaves_pruning_with_mallows(leaves):
  #   quantile=jnp.percentile(jnp.abs(jnp.array(leaves)), 100-pruning_amount)
  #   mask=jax.tree_util.tree_map(lambda x: get_mask(x, quantile), leaves)
  #   mask_p=mallows_model(mask, pruning_amount)
  #   return jax.tree_util.tree_map(lambda x,y: x*y , leaves, mask_p)

  
  def get_mask(x, quantile):
    return  jnp.where(jnp.abs(x) > quantile, jnp.ones_like(x), jnp.zeros_like(x))
  

  return pruning_fn

def _value_and_clipped_grad_single_sample(     #no pruning
    forward_fn: LossFn,
    clipping_fn: ClippingFn,
    clipping_error_fn: ClippingFn,
) -> GradFn:
  """Create a function that computes a clipped gradient for a single sample.

  Args:
    forward_fn: function that should be differentiated. It is expected to have
      the following signature:
      `loss, aux = forward_fn(params, inputs, network_state, rngh_key)`.
    clipping_fn: clipping function to apply to the gradient.

  Returns:
    Function that computes the gradient for a single sample and clips it.
  """

  def grad_fn(
      params: Params,
      inputs: Inputs,
      network_state: ModelState,
      rng: chex.PRNGKey,
      mask: Params,
  ) -> Tuple[Tuple[Loss, Aux], Tuple[GradParams, Aux]]:
    # Add a batch-size dimension.
    inputs_expanded = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(x, axis=0),
        inputs,
    )

    # Compute the gradient.
    out, grad = jax.value_and_grad(forward_fn, has_aux=True)(
        params, inputs_expanded, network_state, rng)
    
    pruned_grad = jax.tree_util.tree_map(lambda x,y: x * y, grad, mask)
    # pruned_grad = grad

    # error = jax.tree_util.tree_map(lambda x,y: x * (1-y), grad, mask)
    error = grad

    # Apply the clipping function
    return out, clipping_fn(pruned_grad), clipping_fn(error)

  return grad_fn

def _value_and_clipped_pruned_grad_single_sample(  ##
    forward_fn: LossFn,
    clipping_fn: ClippingFn,
    pruning_fn: PruningFn,
) -> GradFn:
  """Create a function that computes a clipped and pruned gradient for a single sample.

  Args:
    forward_fn: function that should be differentiated. It is expected to have
      the following signature:
      `loss, aux = forward_fn(params, inputs, network_state, rngh_key)`.
    clipping_fn: clipping function to apply to the gradient.
    pruning_fn

  Returns:
    Function that computes the gradient for a single sample and clips it.
  """

  def grad_fn(
      params: Params,
      inputs: Inputs,
      network_state: ModelState,
      rng: chex.PRNGKey,
  ) -> Tuple[Tuple[Loss, Aux], Tuple[GradParams, Aux]]:
    # Add a batch-size dimension.
    inputs_expanded = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(x, axis=0),
        inputs,
    )

    # Compute the gradient.
    out, grad = jax.value_and_grad(forward_fn, has_aux=True)(
        params, inputs_expanded, network_state, rng)

    # Apply the clipping function
    return out, pruning_fn(clipping_fn(grad))

  return grad_fn


def value_and_clipped_grad_loop(
    forward_fn: LossFn,
    clipping_fn: ClippingFn,
) -> GradFn:
  """Create a function that computes grads clipped per example using a loop.

  Args:
    forward_fn: function that should be differentiated. It is expected to have
      the following signature:
      `loss, aux = forward_fn(params, inputs, network_state, rngh_key)`.
      If looping adds a leading dimension to an entry of `aux`, that entry will
      get automatically averaged after the loop. Other entries of `aux` are left
      untouched after the loop.
    clipping_fn: clipping function to apply to every per-example gradient before
      those get averaged.

  Returns:
    Function that clips gradient per-example and average them.
  """

  grad_fn_single_sample = _value_and_clipped_grad_single_sample(
      forward_fn=forward_fn,
      clipping_fn=clipping_fn,
  )

  grad_fn_vectorized = jax.vmap(
      grad_fn_single_sample,
      in_axes=(None, 0, None, None),
  )

  accumulator = grad_clipping_utils.LoopAccumulator(
      grad_clipping_utils.ShapeEvaluator(
          forward_fn, clipping_fn, grad_fn_vectorized),
  )

  def grad_fn(
      params: Params,
      inputs: Inputs,
      network_state: ModelState,
      rng: chex.PRNGKey,
  ) -> Tuple[Tuple[Loss, Aux], Tuple[GradParams, Aux]]:

    batch_size = jax.tree_leaves(inputs)[0].shape[0]

    if batch_size == 1:
      inputs_0 = jax.tree_util.tree_map(lambda x: x[0], inputs)
      return grad_fn_single_sample(
          params, inputs_0, network_state, rng)

    def body(value_and_grad, i):
      inputs_i = jax.tree_util.tree_map(lambda x: x[i], inputs)
      value_and_grad_i = grad_fn_single_sample(
          params, inputs_i, network_state, rng)
      value_and_grad = accumulator.accumulate(
          value_and_grad, value_and_grad_i, i, batch_size)
      return value_and_grad, None

    # We only need to know the shape and dtype for the initialization, so we
    # pass the arguments through `_placeholder_like` to make that clear.
    placeholder_args = _placeholder_like(params, inputs, network_state, rng)
    value_and_grad = accumulator.initialize(*placeholder_args)

    # Actually perform the loop.
    value_and_grad, _ = jax.lax.scan(
        body, value_and_grad, jnp.arange(batch_size))
    return value_and_grad

  return grad_fn


def value_and_clipped_grad_vectorized(
    forward_fn: LossFn,
    clipping_fn: ClippingFn,
    clipping_error_fn: ClippingFn=None,
) -> GradFn:
  """Create a function that computes grads clipped per example using vmapping.

  Args:
    forward_fn: function that should be differentiated. It is expected to have
      the following signature:
      `loss, aux = forward_fn(params, inputs, network_state, rngh_key)`.
      If vmapping adds a leading dimension to an entry of `aux`, that entry will
      get automatically averaged after vmapping. Other entries of `aux` are left
      untouched after vmapping.
    clipping_fn: clipping function to apply to every per-example gradient before
      those get averaged.

  Returns:
    Function that clips gradient per-example and average them.
  """
  # if pruning_fn is not None:
  #   grad_fn_single_sample = _value_and_clipped_pruned_grad_single_sample(  ##
  #       forward_fn=forward_fn,
  #       clipping_fn=clipping_fn,
  #       pruning_fn=pruning_fn,
  #   )
  # else:
  grad_fn_single_sample = _value_and_clipped_grad_single_sample(  ##
      forward_fn=forward_fn,
      clipping_fn=clipping_fn,
      clipping_error_fn=clipping_error_fn,
  )


  grad_fn_vectorized = jax.vmap(
      grad_fn_single_sample,
      in_axes=(None, 0, None, None, None),
  )

  vmap_reducer = grad_clipping_utils.VmapReducer(    ###???
      grad_clipping_utils.ShapeEvaluator(
          forward_fn, clipping_fn, clipping_error_fn, grad_fn_vectorized),
  )

  def grad_fn(
      params: Params,
      inputs: Inputs,
      network_state: ModelState,
      rng: chex.PRNGKey,
      mask: Params,
  ) -> Tuple[Tuple[Loss, Aux], Tuple[GradParams, Aux]]:

    # Compute vectorized outputs and clipped gradients.
    # print(inputs.shape)
    vectorized_value_and_grad = grad_fn_vectorized(
        params, inputs, network_state, rng, mask)

    # We only need to know the shape and dtype for the reduction, so we pass
    # the arguments through `_placeholder_like` to make that clear.
    placeholder_args = _placeholder_like(params, inputs, network_state, rng, mask)
    return vmap_reducer.reduce(vectorized_value_and_grad, *placeholder_args)

  return grad_fn


def datalens_pruning(
  pruning_k: chex.Array,
) -> PruningFn:
  def pruning_fn(grad: GradParams) -> Tuple[GradParams, Aux]:
    tree_value, tree_def=jax.tree_flatten(grad)
    tree_value=map(leaves_pruning, tree_value)
    # tree_value=jax.vmap(lambda x: leaves_pruning(x))(tree_value)
    tree_output=jax.tree_unflatten(tree_def, tree_value)
    return tree_output


  def leaves_pruning(leaves):
    grad_norm = optax.global_norm(leaves)**2
    target_norm = grad_norm * (1-pruning_k)
    cumnorm=jnp.cumsum(jnp.sort((jnp.abs(leaves)).flatten()) ** 2)
    k_count=jnp.sum(cumnorm < target_norm) + 1   ##maybe error if k=1 
    quantile=jnp.percentile(jnp.abs(leaves), k_count * 100 / jnp.size(cumnorm)  )
    return jax.tree_util.tree_map(lambda x: x * get_mask(x, quantile), leaves)

  def get_mask(x, quantile):
    return  jnp.where(jnp.abs(x) > quantile, jnp.ones_like(x), jnp.zeros_like(x))

  return pruning_fn





