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

"""The updater computes and applies the update.

Typical usage:
  # The updater requires a (haiku) init function, a forward function and a
  # batching instance.
  updater = updater.Updater(
        batching=batching,  # see `batching.py`
        train_init=train_init,  # init function of a haiku model
        forward=train_forward,  # see `forward.py`
        ...
  )

  ...

  # Initialize model and optimizer (pmapped).
  params, network_state, opt_state = updater.init(inputs, rng_key)

  # Apply update (pmapped).
  params, network_state, opt_state, stats = updater.update(
      params=params,
      network_state=network_state,
      opt_state=opt_state,
      global_step=global_step,
      inputs=inputs,
      rng=rng,
  )
"""

# from ast import Try
from cProfile import label
import functools
# from tkinter import S
from typing import Any, Dict, Mapping, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp

from jax_privacy.src.training import batching as batching_module
from jax_privacy.src.training import grad_clipping, mallows_model_256
from jax_privacy.src.training import optim
from jax_privacy.src.training import cos_sim
from jaxline import utils
import optax
from typing import Callable

import ipdb

Model = hk.TransformedWithState
InitFn = Any
ForwardFn = Any


Aux = chex.ArrayTree
Params = chex.ArrayTree
GradParams = Params
PruningFn = Callable[[GradParams], Tuple[GradParams, Aux]]

class Updater:
  """Defines and applies the update, potentially in parallel across devices."""

  def __init__(
      self,
      *,
      batching: batching_module.VirtualBatching,
      train_init: InitFn,
      forward: ForwardFn,
      noise_std_relative: Optional[chex.Numeric],
      clipping_norm: Optional[chex.Numeric],
      rescale_to_unit_norm: bool,
      weight_decay: Optional[chex.Numeric],
      train_only_layer: Optional[str],
      optimizer_name: str,
      optimizer_kwargs: Optional[Mapping[str, Any]],
      lr_init_value: chex.Numeric,
      lr_decay_schedule_name: Optional[str],
      lr_decay_schedule_kwargs: Optional[Mapping[str, Any]],
      log_snr_global: bool = False,
      log_snr_per_layer: bool = False,
      log_grad_clipping: bool = False,
      log_grad_alignment: bool = False,
      per_example_pruning_amount: Optional[chex.Numeric],
      batch_pruning_amount: Optional[chex.Numeric],
      pruning_eps_step: Optional[chex.Numeric],
      model_type: Optional[str],
      # paramsNum: Optional[chex.Numeric],

      max_step: Optional[chex.Numeric],
      batch_pruning_method = "Random",   #TopK Random
      error_sigma: Optional[chex.Numeric],

      datalens_pruning: bool = False,
      datalens_k: Optional[chex.Numeric] = 0.8,


  ):
    """Initializes the updater.

    Args:
      batching: virtual batching that allows to use 'virtual' batches across
        devices and steps.
      train_init: haiku init function to initialize the model.
      forward: function that defines the loss function and metrics.
      noise_std_relative: standard deviation of the noise to add to the average
         of the clipped gradient to make it differentially private. It will be
         multiplied by `clipping_norm / batch_size` before the noise gets
         actually added.
      clipping_norm: clipping-norm for the per-example gradients (before
        averaging across the examples of the mini-batch).
      rescale_to_unit_norm: whether each clipped per-example gradient gets
        multiplied by `1 / clipping_norm`, so that the update is normalized.
        When enabled, the noise standard deviation gets adjusted accordingly.
      weight_decay: whether to apply weight-decay on the parameters of the model
        (mechanism not privatized since it is data-independent).
      train_only_layer: if set to None, train on all layers of the models. If
        specified as a string, train only layer whose name is an exact match
        of this string.
      optimizer_name: name of the optax optimizer to use.
      optimizer_kwargs: keyword arguments passed to optax when creating the
        optimizer (except for the learning-rate, which is handled in this
        class).
      lr_init_value: initial value for the learning-rate.
      lr_decay_schedule_name: if set to None, do not use any schedule.
        Otherwise, identifier of the optax schedule to use.
      lr_decay_schedule_kwargs: keyword arguments for the optax schedule being
        used.
      log_snr_global: whether to log the Signal-to-Noise Ratio (SNR) globally
        across layers, where the SNR is defined as:
        ||non_private_grads||_2 / ||noise||_2.
      log_snr_per_layer: whether to log the Signal-to-Noise Ratio (SNR) per
        layer, where the SNR is defined as:
        ||non_private_grads||_2 / ||noise||_2.
      log_grad_clipping: whether to log the proportion of per-example gradients
        that get clipped at each iteration.
      log_grad_alignment: whether to compute the gradient alignment: cosine
        distance between the differentially private gradients and the
        non-private gradients computed on the same data.
    """
    self.random_key=jax.random.PRNGKey(3407)
    self.quantile=None
    self.batching = batching
    self._train_init = train_init
    self._forward = forward


    ##datalens
    self._datalens_pruning = datalens_pruning
    self._datalens_k=datalens_k

    self._clipping_norm = clipping_norm
    self._noise_std_relative = noise_std_relative
    self._rescale_to_unit_norm = rescale_to_unit_norm
    self._weight_decay = weight_decay
    self._train_only_layer = train_only_layer

    self._optimizer_name = optimizer_name
    self._optimizer_kwargs = optimizer_kwargs
    self._lr_init_value = lr_init_value
    self._lr_decay_schedule_name = lr_decay_schedule_name
    self._lr_decay_schedule_kwargs = lr_decay_schedule_kwargs

    self._log_snr_global = log_snr_global
    self._log_snr_per_layer = log_snr_per_layer
    self._log_grad_clipping = log_grad_clipping
    self._log_grad_alignment = log_grad_alignment

    self._per_example_pruning_amount = per_example_pruning_amount
    self._batch_pruning_amount= batch_pruning_amount
    self._batch_pruning_method=batch_pruning_method
    self.paramsNum=None
    self.group_num=None
    self.change_num_list = None #jnp.load('/home/jungang/jax_privacy/jax_privacy/src/training/change_num_list.npy')
    self.steps = 0
    self.model_type = model_type

    self.error = None
    self.error_clip_norm = 0.1
    self.error_std = error_sigma
    self.error_norm = 0.0
    self.clipped_error_norm = 0.0
    self.mask = None
    self.random_split = False
    

    self.max_step=max_step

    self._random_key=jax.random.PRNGKey(42)
   

    self.batch_pruningFn=None

    if (clipping_norm in (float('inf'), None) and
        rescale_to_unit_norm):
      raise ValueError('Cannot rescale to unit norm without clipping.')
    elif clipping_norm in (float('inf'), None):
      # We can compute standard gradients.
      self._using_clipped_grads = False
      self.value_and_clipped_grad = functools.partial(
          jax.value_and_grad, has_aux=True)
    else:
      self._using_clipped_grads = True
      error_clipping_fn = grad_clipping.global_clipping(
              clipping_norm=self.error_clip_norm,
              rescale_to_unit_norm=rescale_to_unit_norm,
          )
      self.value_and_clipped_grad = functools.partial(
          grad_clipping.value_and_clipped_grad_vectorized,
          clipping_fn=grad_clipping.global_clipping(
              clipping_norm=clipping_norm,
              rescale_to_unit_norm=rescale_to_unit_norm,
          ),
          clipping_error_fn=error_clipping_fn,
          ##
          # pruning_fn=grad_clipping.pruning(
          #     pruning_amount=20,    
          # ),
      )
    self.pruning_eps_step = pruning_eps_step

  def _regularization(self, params: chex.ArrayTree) -> chex.Array:
    l2_loss = optim.l2_loss(params)
    return self._weight_decay * l2_loss, l2_loss

  def _is_trainable(
      self,
      layer_name: str,
      unused_parameter_name: str,
      unused_parameter_value: chex.Array,
  ) -> bool:
    if self._train_only_layer:
      return layer_name == self._train_only_layer
    else:
      return True

  def init(
      self,
      *,
      inputs: chex.ArrayTree,
      rng_key: chex.PRNGKey,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]:
    """Initialization function."""
    return self._pmapped_init(inputs, rng_key)

  @functools.partial(jax.pmap, static_broadcasted_argnums=0, axis_name='i')
  def _pmapped_init(
      self,
      inputs: chex.ArrayTree,
      rng_key: chex.PRNGKey,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree]:
    """Initialization function (to be pmapped)."""
    # print('label_2')
    # print('inputsss_2', inputs)
    params, network_state = self._train_init(rng_key, inputs)

    trainable_params, unused_frozen_params = hk.data_structures.partition(
        self._is_trainable, params)

    opt_init, _ = optim.optimizer(
        optimizer_name=self._optimizer_name,
        every_k_schedule=self.batching.apply_update_every,
        optimizer_kwargs=self._optimizer_kwargs,
        learning_rate=0.0,
    )
    opt_state = opt_init(trainable_params)
    return params, network_state, opt_state

  def update(
      self,
      *,
      params: chex.ArrayTree,
      network_state: chex.ArrayTree,
      opt_state: chex.ArrayTree,
      global_step: chex.Array,
      inputs: chex.ArrayTree,
      rng: chex.PRNGKey,
      #mode: chex.Array,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree, Any]:
    """Perform the pmapped update."""
    # The function below is p-mapped, so arguments must be provided without name
    # and in the right order, hence why we define this method, which has to be
    # called with named arguments in order to avoid any mistake.
    # opt_update(global_step, grad(loss)(params, batch), opt_state)
    
    
    return self._pmapped_update(
        params,
        network_state,
        opt_state,
        global_step,
        inputs,
        rng,
        utils.host_id_devices_for_rng(),
    )
  
  def batch_pruning_top_k(  
    self,
    batch_pruning_amount: chex.Array,
  ) -> PruningFn:

    def pruning_fn(grad_mask: GradParams) -> Tuple[GradParams, Aux]:
      # tree_value, tree_def=jax.tree_util.tree_flatten(grad)
      tree_mask_value, tree_mask_def=jax.tree_util.tree_flatten(grad_mask)
      # print('mask_leaves', tree_mask_def)
      tree_mask_value= map(leaves_split, tree_mask_value)
      tree_output=jax.tree_util.tree_unflatten(tree_mask_def, tree_mask_value)
      # del tree_value, tree_mask_value, tree_def, tree_mask_def
      return tree_output

    def leaves_split(mask_leaves):
      mask = mask_leaves.reshape(-1)
      quantile=jnp.percentile(jnp.abs(mask), 100-batch_pruning_amount)
      mask = jnp.where(jnp.abs(mask)> quantile, 1, 0)
      # mask = self.mallows_mode.model(mask, batch_pruning_amount)
      return mask.reshape(mask_leaves.shape)

    return pruning_fn

  def batch_pruning_topk_split(  
    self,
    batch_pruning_amount: chex.Array,
    theta: chex.Array,
  ) -> PruningFn:

    def pruning_fn(grad, grad_mask: GradParams) -> Tuple[GradParams, Aux]:
      # tree_value, tree_def=jax.tree_util.tree_flatten(grad)
      tree_mask_value, tree_mask_def=jax.tree_util.tree_flatten(grad_mask)
      # print('mask_leaves', tree_mask_def)
      tree_mask_value= map(leaves_split, tree_mask_value)
      tree_output=jax.tree_util.tree_unflatten(tree_mask_def, tree_mask_value)
      # del tree_value, tree_mask_value, tree_def, tree_mask_def
      return jax.tree_util.tree_map(lambda x, y: x*y, tree_output, grad), jax.tree_util.tree_map(lambda x, y: x*y, tree_output, grad_mask)

    def leaves_split(mask_leaves):
      mask = mask_leaves.reshape(-1)
      mask = self.mallows_mode.model(mask, batch_pruning_amount)
      return mask.reshape(mask_leaves.shape)

    return pruning_fn

  def batch_pruning_random(  
    self,
    batch_pruning_amount: chex.Array,
  ) -> PruningFn:

    def pruning_fn(grad: GradParams) -> Tuple[GradParams, Aux]:
      tree_value, tree_def=jax.tree_util.tree_flatten(grad)
      tree_value=map(leaves_pruning,tree_value)
      tree_output=jax.tree_util.tree_unflatten(tree_def,tree_value)
      return tree_output

    def leaves_pruning(leaves):
      random_grad = jax.random.normal(self.random_key,jnp.shape(leaves))
      quantile=jnp.percentile(jnp.abs(random_grad), 100-batch_pruning_amount)
      # return jax.tree_util.tree_map(lambda x: x * get_mask(random_grad, quantile), leaves)
      return jnp.where(jnp.abs(random_grad)> quantile, 1, 0)
  
    def get_mask(x, quantile):
      return jnp.where(jnp.abs(x)> quantile, 1, 0)

    return pruning_fn

  def batch_pruning_random_split(  
    self,
    batch_pruning_amount: chex.Array,
  ) -> PruningFn:

    def pruning_fn(grad: GradParams) -> Tuple[GradParams, Aux]:
      tree_value, tree_def=jax.tree_util.tree_flatten(grad)
      tree_value=map(leaves_split,tree_value)
      tree_output=jax.tree_util.tree_unflatten(tree_def,tree_value)
      return tree_output

    # def leaves_pruning(leaves):
    #   random_grad = jax.random.normal(self.random_key,jnp.shape(leaves))
    #   quantile=jnp.percentile(jnp.abs(random_grad), 100-batch_pruning_amount)
    #   # return jax.tree_util.tree_map(lambda x: x * get_mask(random_grad, quantile), leaves)
    #   return jnp.where(jnp.abs(random_grad)> quantile, 1, 0)
  
    def leaves_split(mask_leaves):
      mask = mask_leaves.reshape(-1)
      mask = self.random_mode.model(mask, batch_pruning_amount)
      return mask.reshape(mask_leaves.shape)

    return pruning_fn


  @functools.partial(jax.pmap, static_broadcasted_argnums=0, axis_name='i')
  def _pmapped_update(
      self,
      params: chex.ArrayTree,
      network_state: chex.ArrayTree,
      opt_state: chex.ArrayTree,
      global_step: chex.Array,
      inputs: chex.ArrayTree,
      rng: chex.PRNGKey,
      host_id: Optional[chex.Array],
      #mode: chex.Array,
  ) -> Tuple[chex.ArrayTree, chex.ArrayTree, chex.ArrayTree, Any]:
    """Updates parameters."""
    # Note on rngs:
    # - rng is common across replicas thanks to config.random_train,
    # - therefore rng_common also common across replicas,
    # - rng_device is specialised per device (for independent randonmness).
    rng_tmp, rng_common = jax.random.split(rng)
    rng_device = utils.specialize_rng_host_device(
        rng_tmp, host_id, axis_name='i', mode='unique_host_unique_device')

    # Save the initial network state before it gets updated by a forward pass.
    initial_network_state = network_state

    # The update step is logged in the optimizer state (by optax.MultiSteps)
    #  under the name of 'gradient_step'.
    update_step = opt_state.gradient_step

    # Potentially split params between trainable parameters and frozen
    # parameters. Trainable parameters get updated, while frozen parameters do
    # not.
    params, frozen_params = hk.data_structures.partition(
        self._is_trainable, params)

    # Compute clipped-per-example gradients of the loss function (w.r.t. the
    # trainable parameters only).
    def get_sum(mask):
        leaves_value, structure = jax.tree_util.tree_flatten(mask)
        sum = 0
        for leave in leaves_value:
          sum += jnp.sum(leave)
        return sum
    
    theta = 0
    if self._batch_pruning_method == 'TopK_first':
      forward = functools.partial(self._forward, frozen_params=frozen_params)

      device_clean_grads, unused_aux = jax.grad(forward, has_aux=True)(
          params, inputs, network_state, rng_device)

      avg_clean_grads = jax.lax.pmean(device_clean_grads, axis_name='i')
      del device_clean_grads, unused_aux
      def get_netNum(params):
        leaves_value, structure = jax.tree_util.tree_flatten(params)
        group_num = 0
        num = 0
        for leave in leaves_value:
          num += jnp.size(leave)
          # print(leave.shape)
          group_num += jnp.ceil(jnp.size(leave) / 256.)
        return num, group_num
      
      if self.paramsNum is None:
        self.paramsNum, self.group_num = get_netNum(params)
        print('\n==> Number of model:', self.paramsNum)
      # # list_len = 100#jnp.array((paramsNum/256+1), int)
      # print("************************", self.pruning_eps_step)
      linear_pruning_amount = 99.9 - (99.9 - self._batch_pruning_amount)*global_step/self.max_step
      # print("************************", linear_pruning_amount)
      if self.model_type == 'resnet18':
        eps_step = self.pruning_eps_step / 3 + 4 * self.pruning_eps_step / 3 * global_step/self.max_step
        theta = eps_step / (self.paramsNum  * 2 * jnp.where(linear_pruning_amount > 50, 100 - linear_pruning_amount, linear_pruning_amount) / 100) 
      else:
        theta = self.pruning_eps_step / (self.paramsNum  * 2 * jnp.where(linear_pruning_amount > 50, 100 - linear_pruning_amount, linear_pruning_amount) / 100)
      # theta = 0.1
      self.mallows_mode=mallows_model_256.MallowsModel(256, theta, self._batch_pruning_amount)
      self.batch_pruningFn = self.batch_pruning_topk_split(linear_pruning_amount, theta)

      grad_pruned, _=self.batch_pruningFn(avg_clean_grads, avg_clean_grads)
      self.mask = jax.tree_util.tree_map(lambda x: jnp.where(x == 0, 0, 1), grad_pruned)
      mask_norm = get_sum(self.mask)
      mask_ratio = mask_norm / self.paramsNum
      # ipdb.set_trace()
    elif self._batch_pruning_method == 'Random':
      forward = functools.partial(self._forward, frozen_params=frozen_params)

      device_clean_grads, unused_aux = jax.grad(forward, has_aux=True)(
          params, inputs, network_state, rng_device)

      avg_clean_grads = jax.lax.pmean(device_clean_grads, axis_name='i')
      del device_clean_grads, unused_aux
      if self.random_split:
        linear_pruning_amount = 99.9 - (99.9 - self._batch_pruning_amount)*global_step/self.max_step
        self.random_mode=mallows_model_256.RandomModel()
        self.batch_pruningFn = self.batch_pruning_random_split(linear_pruning_amount)
      else:
        self.batch_pruningFn = self.batch_pruning_random(99.9 - jnp.exp(jnp.log(99.9 - self._batch_pruning_amount)*global_step/self.max_step))   
      self.mask=self.batch_pruningFn(avg_clean_grads)
      mask_norm = get_sum(self.mask)
      print('pruning finished')
    else:
      forward = functools.partial(self._forward, frozen_params=frozen_params)
      device_clean_grads, unused_aux = jax.grad(forward, has_aux=True)(
          params, inputs, network_state, rng_device)
      avg_clean_grads = jax.lax.pmean(device_clean_grads, axis_name='i')
      del device_clean_grads, unused_aux
      self.mask = None




    if self._datalens_pruning:
      (loss, (network_state, metrics,
        loss_vector)), device_grads = self.value_and_clipped_grad(forward, pruning_fn=grad_clipping.datalens_pruning(self._datalens_k))(
        params, inputs, network_state, rng_device, self.mask)
      print('datalens works')
      mask_norm = 0
    elif self._batch_pruning_method == 'TopK_first' or  self._batch_pruning_method == 'Random':
      # print('inputshape')
      # print(inputs['images'].shape, inputs['labels'].shape)
      (loss, (network_state, metrics,
        loss_vector)), device_grads = self.value_and_clipped_grad(forward)(
            params, inputs, network_state, rng_device, self.mask)
    else:
      (loss, (network_state, metrics,
        loss_vector)), device_grads = self.value_and_clipped_grad(forward)(
            params, inputs, network_state, rng_device, self.mask)
      mask_norm = 0
            
      print('datalens not works')

    if self._using_clipped_grads:
      device_grads, grad_norms_per_sample, origin_grads = device_grads
      # device_errors, error_norms_per_sample, origin_error = errors
    else:
      grad_norms_per_sample = None
      origin_grads = device_grads


    # Synchronize metrics and gradients across devices.
    loss, metrics, avg_grads, avgori_grads = jax.lax.pmean(
        (loss, metrics, device_grads, origin_grads), axis_name='i')
    # avg_error, avgori_error = jax.lax.pmean(
    #     (device_errors, origin_error), axis_name='i')
    # pruning_fn = self.batch_pruning_top_k(linear_pruning_amount)

    # mask_clipped = pruning_fn(avg_error)
    # mask_ori = pruning_fn(avgori_error)
    # mask_first = pruning_fn(avg_clean_grads)
    # mask_u = jax.tree_util.tree_map(lambda x, y: x*y, mask_first, mask_clipped)
    # mask_check = jax.tree_util.tree_map(lambda x, y: x*y, mask_first, mask_ori)
    # mask_u_ratio = get_sum(mask_u) / mask_norm
    # mask_check_ratio = get_sum(mask_check) / mask_norm
    # mask_distance = get_sum(jax.tree_util.tree_map(lambda x, y: jnp.abs(x-y), mask_first, mask_clipped))
    # mask_check_distance = get_sum(jax.tree_util.tree_map(lambda x, y: jnp.abs(x-y), mask_first, mask_ori))
    
    loss_all = jax.lax.all_gather(loss_vector, axis_name='i')
    loss_vector = jnp.reshape(loss_all, [-1])

    if self._batch_pruning_method == 'TopK_first' or self._batch_pruning_method == 'Random':
      avgori_error = jax.tree_util.tree_map(lambda x, y: x * (1-y), avg_clean_grads, self.mask)
      avg_error = None
      self.error_norm = optax.global_norm(avgori_error)
      # self.clipped_error_norm = optax.global_norm(avg_error)
      self.g_norm = optax.global_norm(avg_clean_grads)
      self.avg_prunedgrad_norm = optax.global_norm(avgori_grads)
      # error_clipped = jnp.max(error_norms_per_sample)
      error_ratio = self.error_norm / self.g_norm
      grad_ratio = self.avg_prunedgrad_norm / self.g_norm
      
    else:
      avg_error = None
      self.error_norm = 0
      self.g_norm = optax.global_norm(avg_grads)
      self.avg_prunedgrad_norm = 0
      error_ratio = 0
      grad_ratio = 1
    ##datalens 
    # datalens_pruningfn=grad_clipping.datalens_pruning(self._datalens_k)
    # avg_grads1=datalens_pruningfn(avg_grads)

    # Compute the regularization and its corresponding gradients. Since those
    # are data-independent, there is no need to privatize / clip them.
    (reg, l2_loss), reg_grads = jax.value_and_grad(
        self._regularization, has_aux=True)(params)

    # Compute the noise scale based on `noise_std_relative`, the batch-size and
    # the clipping-norm. Here the noise is created by being added to a structure
    # of zeros mimicking the gradients structure.
    if self._datalens_pruning:
      noise, std = optim.add_noise_to_grads(
          clipping_norm=self._clipping_norm * self._datalens_k,
          rescale_to_unit_norm=self._rescale_to_unit_norm,
          noise_std_relative=self._noise_std_relative,
          apply_every=self.batching.apply_update_every(global_step),
          total_batch_size=self.batching.batch_size(global_step),
          grads=jax.tree_util.tree_map(jnp.zeros_like, avg_grads),
          rng_key=rng_common,
      )
    else:
      noise, std = optim.add_noise_to_grads(
        clipping_norm=self._clipping_norm,
        rescale_to_unit_norm=self._rescale_to_unit_norm,
        noise_std_relative=self._noise_std_relative,
        apply_every=self.batching.apply_update_every(global_step),
        total_batch_size=self.batching.batch_size(global_step),
        grads=jax.tree_util.tree_map(jnp.zeros_like, avg_grads),
        rng_key=rng_common,
      )

      if avg_error is None:
        error_noise = None
      else:
        error_noise, error_std = optim.add_noise_to_grads(
          clipping_norm=self.error_clip_norm,
          rescale_to_unit_norm=self._rescale_to_unit_norm,
          noise_std_relative=self.error_std,
          apply_every=self.batching.apply_update_every(global_step),
          total_batch_size=self.batching.batch_size(global_step),
          grads=jax.tree_util.tree_map(jnp.zeros_like, avg_error),
          rng_key=rng_common,
        )
        error_noise = jax.tree_util.tree_map(lambda x, y: x * (1-y), error_noise, self.mask)
        
      if self.mask is not None:
        noise = jax.tree_util.tree_map(lambda x, y: x * y, noise, self.mask)
        


    # Compute our 'final' gradients `grads`: these comprise the clipped
    # data-dependent gradients (`avg_grads`), the regularization gradients
    # (`reg_grads`) and the noise to be added to achieved differential privacy
    # (`noise`).
    grads = jax.tree_util.tree_map(
        lambda *args: sum(args),
        avg_grads,
        reg_grads,
        noise,
        # error_noise,
        # avg_error,
    )

    if self._batch_pruning_method == 'TopK':
      #self.batch_pruningFn = self.batch_pruning_top_k(jnp.max(((1-0.9*global_step/self.max_step)*100)[0],50))    #LBPA
      # if mode:
      def get_netNum(params):
        leaves_value, structure = jax.tree_util.tree_flatten(params)
        group_num = 0
        num = 0
        for leave in leaves_value:
          num += jnp.size(leave)
          # print(leave.shape)
          group_num += jnp.ceil(jnp.size(leave) / 256.)
        return num, group_num
      if self.paramsNum is None:
        self.paramsNum, self.group_num = get_netNum(grads)
      # # list_len = 100#jnp.array((paramsNum/256+1), int)
      linear_pruning_amount = 99.9 - (99.9 - self._batch_pruning_amount)*global_step/self.max_step
      theta = self.pruning_eps_step / (self.paramsNum  * jnp.where(linear_pruning_amount > 50, 100 - linear_pruning_amount, linear_pruning_amount) / 100)
      # theta = 0.1
      self.mallows_mode=mallows_model_256.MallowsModel(256, theta, self._batch_pruning_amount)
      self.batch_pruningFn = self.batch_pruning_topk_split(linear_pruning_amount, theta)

      grads, clipped_compress_grads=self.batch_pruningFn(grads, avg_grads)
      # if self.error is not None:
      self.error = jax.tree_util.tree_map(lambda x, y: x - y, avg_grads, clipped_compress_grads)
      self.error_norm = optax.global_norm(self.error)
      # clip_fator = jnp.minimum(1.0, self.error_clip_norm / self.error_norm)
      self.error = jax.tree_util.tree_map(lambda x: x * self.error_clip_norm, self.error)
      if self.error is not None and error_noise is not None:
        error_noise = jax.tree_util.tree_map(lambda x, y: jnp.where(y==0, y, x), error_noise, self.error)
        grads = jax.tree_util.tree_map(
            lambda *args: sum(args),
            grads,
            self.error,
            error_noise,
        )
      # cos = cos_sim.cos_sim(grads, avg_grads)
      print('pruning finished')
    elif self._batch_pruning_method == 'TopK_first' or self._batch_pruning_method == 'Random':
      if self.model_type == 'resnet20':
        cos = cos_sim.cos_sim(grads, avg_clean_grads)
    else:
      # if self._datalens_pruning:
      #   cos = 0
      # else:
      if self.model_type == 'resnet20':
        cos = cos_sim.cos_sim(grads, avg_clean_grads)
      print('Batch pruning skipped')


    # linear_batch_pa


    #print(jax.tree_structure(grads))
     
    #sample_array=jnp.array(grad_norms_per_sample)

    # datalens_pruningfn=grad_clipping.datalens_pruning(self._datalens_k)
    # datalens_pruningfn=grad_clipping.pruning(40)

    # avg_grads1=datalens_pruningfn(avg_grads)

    # grad_leaves=jax.tree_util.tree_leaves(avg_grads)
    # grad_array=grad_leaves[0]
    # grad_leaves=jax.tree_util.tree_leaves(avg_grads1)
    # grad_array1=grad_leaves[0]

    # grad_array=None
    # for i in range(len(grad_leaves)):
    #   if grad_array==None:
    #     grad_array=jnp.array(grad_leaves[i])
    #   else:
    #     grad_array=jnp.append(grad_array,jnp.array(grad_leaves[i]))

    # Compute the learning-rate according to its schedule. Note that the
    # schedule evolves with `update_step` rather than `global_step` since the
    # former accounts for the fact that gradient smay be accumulated over
    # multiple global steps.
    learning_rate = optim.learning_rate_schedule(
        update_step=update_step,
        init_value=self._lr_init_value,
        decay_schedule_name=self._lr_decay_schedule_name,
        decay_schedule_kwargs=self._lr_decay_schedule_kwargs,
    )

    # Create an optimizer that will only apply the update every
    # `k=self.batching.apply_update_every` steps, and accumulate gradients
    # in-between so that we can use a large 'virtual' batch-size.
    _, opt_apply = optim.optimizer(
        learning_rate=learning_rate,
        optimizer_name=self._optimizer_name,
        optimizer_kwargs=self._optimizer_kwargs,
        every_k_schedule=self.batching.apply_update_every,
    )

    # Log all relevant statistics in a dictionary.
    scalars = dict(
        learning_rate=learning_rate,
        noise_std=std,
        train_loss=loss,
        train_loss_mean=jnp.mean(loss_vector),
        train_loss_min=jnp.min(loss_vector),
        train_loss_max=jnp.max(loss_vector),
        train_loss_std=jnp.std(loss_vector),
        train_loss_median=jnp.median(loss_vector),
        reg=reg,
        batch_size=self.batching.batch_size(global_step),
        data_seen=self.batching.data_seen(global_step),
        update_every=self.batching.apply_update_every(global_step),
        l2_loss=l2_loss,
        train_obj=(reg + loss),
        grads_norm=optax.global_norm(grads),
        update_step=update_step,
        error_norm=self.error_norm,
        noise_sigma=self._noise_std_relative,
        # error_clipped_norm=self.clipped_error_norm,
        # error_clipped=error_clipped,
        avg_grads_norm=self.g_norm,
        avg_clipped_norm=optax.global_norm(avg_grads),
        mask_norm = mask_norm,
        # mask_u_ratio=mask_u_ratio,
        # error_sigma=self.error_std,
        error_ratio=error_ratio,
        grad_ratio=grad_ratio,
        theta=theta,
        # mask_check_ratio=mask_check_ratio,
        # mask_distance=mask_distance,
    )

    scalars.update(metrics)
    #scalars.update(grad_e9)
    # Possibly log additional statistics from the gradient.
    scalars.update(self._compute_gradient_stats(
        params=params,
        frozen_params=frozen_params,
        inputs=inputs,
        rng_device=rng_device,
        network_state=network_state,
        initial_network_state=initial_network_state,
        grads=grads,
        reg_grads=reg_grads,
        avg_grads=avg_grads,
        grad_norms_per_sample=grad_norms_per_sample,
        noise=noise,
    ))

    # Perform the update on the model parameters (no-op if this step is meant to
    # accumulate gradients rather than performing the model update).
    updates, opt_state = opt_apply(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Merge the updated parameters with the parameters that are supposed to
    # remain frozen during training.
    new_params = hk.data_structures.merge(new_params, frozen_params)

    if self.model_type == 'resnet20':
      return new_params, network_state, opt_state, scalars, (cos, error_ratio)
    else:
      return new_params, network_state, opt_state, scalars

  def _compute_grad_alignment(
      self,
      params: chex.ArrayTree,
      frozen_params: chex.ArrayTree,
      inputs: chex.ArrayTree,
      network_state: chex.ArrayTree,
      rng_device: chex.PRNGKey,
      grads: chex.ArrayTree,
      reg_grads: chex.ArrayTree,
  ) -> chex.Array:
    """Compute alignment between grads used and 'clean' grads."""

    # Compute (non-clipped) gradients w.r.t. trainable parameters.
    forward = functools.partial(self._forward, frozen_params=frozen_params)
    device_clean_grads, unused_aux = jax.grad(forward, has_aux=True)(
        params, inputs, network_state, rng_device)

    avg_clean_grads = jax.lax.pmean(device_clean_grads, axis_name='i')
    
    #sample num=per_device_per_step
    #1e7+ grad

    # gradients: normalized accumulated gradients + reg gradient
    clean_grads = jax.tree_util.tree_map(
        lambda x1, x2: x1 + x2,
        avg_clean_grads,
        reg_grads,
    )

    return optim.cosine_distance(grads, clean_grads)

  def _compute_gradient_stats(
      self,
      *,
      params: chex.ArrayTree,
      frozen_params: chex.ArrayTree,
      inputs: chex.ArrayTree,
      rng_device: chex.PRNGKey,
      network_state: chex.ArrayTree,
      initial_network_state: chex.ArrayTree,
      grads: chex.ArrayTree,
      reg_grads: chex.ArrayTree,
      avg_grads: chex.ArrayTree,
      grad_norms_per_sample: chex.Array,
      noise: chex.ArrayTree,
  ) -> Dict[str, Any]:
    """Compute various gradient statistics for logging."""
    del network_state  # unused
    stats = {}
    # Log Signal-to-Noise Ratio.
    if self._log_snr_global:
      stats['snr_global'] = (
          optax.global_norm(avg_grads) / optax.global_norm(noise))

    if self._log_snr_per_layer:
      signal_to_noise_per_layer = jax.tree_util.tree_map(
          lambda x1, x2: jnp.linalg.norm(x1) / jnp.linalg.norm(x2),
          avg_grads,
          noise,
      )
      for mod_name, name, value in hk.data_structures.traverse(
          signal_to_noise_per_layer):
        stats.update({f'snr_{mod_name}_{name}': value})

    if self._log_grad_clipping:
      if self._clipping_norm in (None, float('inf')):
        stats.update(grads_clipped=0.0)
      else:
        grads_clipped = jnp.mean(jnp.greater(
            grad_norms_per_sample, self._clipping_norm))
        stats.update(
            grads_clipped=grads_clipped,
            grad_norms_before_clipping_mean=jnp.mean(grad_norms_per_sample),
            grad_norms_before_clipping_median=jnp.median(grad_norms_per_sample),
            grad_norms_before_clipping_min=jnp.min(grad_norms_per_sample),
            grad_norms_before_clipping_max=jnp.max(grad_norms_per_sample),
            grad_norms_before_clipping_std=jnp.std(grad_norms_per_sample),
        )

    if self._log_grad_alignment:
      grad_alignment = self._compute_grad_alignment(params, frozen_params,
                                                    inputs,
                                                    initial_network_state,
                                                    rng_device, grads,
                                                    reg_grads)
      stats.update(grad_alignment=grad_alignment)

    return stats
