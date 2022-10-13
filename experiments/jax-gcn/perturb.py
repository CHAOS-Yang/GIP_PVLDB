import numpy as np
from typing import Callable, Tuple
from jax_privacy.src import accounting
from jax_privacy.src.training import batching
import jax
import jax.numpy as jnp
import optax
import chex
import mallows_model_256

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

_epsilon = 1.0
_delta = 1e-3
_l2_clip_value = 1e-2
_total_epochs = 60
_pruning_method = None
_index_noise_weight = 0.01

class GauMechanism(object):
    def __init__(self, epsilon=_epsilon, delta=_delta,
                 l2_clip_value=_l2_clip_value,
                 total_epochs=_total_epochs,
                 pruning_method=_pruning_method,
                 index_noise_weight=_index_noise_weight ):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta
        self._sample_rate_q = 1
        self._noise_iter = total_epochs
        self._clip_value = l2_clip_value
        self.batch_pruning_method = pruning_method
        self._sigma = 10
        self._num_training = 140
        self.index_noise_weight = index_noise_weight


        self.batching = batching.VirtualBatching(
        batch_size_init=self._num_training,
        batch_size_per_device_per_step=self._num_training,
        scale_schedule=None,
        )
        self.accountant = accounting.Accountant(
            clipping_norm=self._clip_value,
            std_relative=self._sigma,
            dp_epsilon=self._epsilon,
            dp_delta=self._delta,
            batching=self.batching,
            num_samples=self._num_training,
        )

        # self._max_num_updates = self.accountant.compute_max_num_updates()
        if self.batch_pruning_method=="TopK":
            pruning_eps = self._epsilon * self.index_noise_weight
            # pruning_eps_step = pruning_eps / self._max_num_updates / self.config.training.batch_size.init_value * self.num_training_samples
            self.accountant._dp_epsilon=self._epsilon * (1 - self.index_noise_weight)
            self._sigma=self.accountant.compute_target_sigma(self._noise_iter)
            
        else:
            pruning_eps_step = None
            self._sigma=self.accountant.compute_target_sigma(self._noise_iter)
            # sigma=self.config.training.dp.noise.std_relative
        # print('noise_std=', self._sigma)

    def current_eps(self, num_updates):
        return self.accountant.compute_current_epsilon(int(num_updates))


    def generate_noise(self, grad, num_supports):
        noiselist = []
        rng_key = jax.random.PRNGKey(42)
        rng_key, _ = jax.random.split(rng_key)
        for l in range(len(grad)//num_supports):
            l2_sen = self._clip_value
            w,b = grad[l*num_supports]
            shape = w.shape
            noise_w = jax.random.normal(rng_key, shape) * l2_sen * self._sigma
            # noiselist.append(noise)
            if b is not None:
                shape = b.shape
                noise_b = jax.random.normal(rng_key, shape) * l2_sen * self._sigma
                noiselist.append((noise_w, noise_b))
            else:
                noiselist.append((noise_w, None))

        # print(noiselist)
        return noiselist


def noiseGen(epsilon, delta, train_epochs, clip_value, pruning_method, index_noise_weight):
    noiseGen = GauMechanism(epsilon=epsilon, delta=delta, total_epochs=train_epochs, l2_clip_value=clip_value,pruning_method=pruning_method, index_noise_weight=index_noise_weight)
    return noiseGen

def safe_div(
    numerator: chex.Array,
    denominator: chex.Array,
    eps: chex.Numeric = 1e-10,
) -> ClippingFn:
  """Numerically safe division."""
  return numerator / (denominator + eps)

def get_netNum(params):
    group_num = 0
    num = 0
    for layers in params:
        w,b = layers
        num += jnp.size(w)
        group_num += jnp.ceil(w.size /  256.)
    # print(type(num))
    # print(type(group_num))
    return num, group_num

def perturb_grad(grad, clip_value, pruning_key, epsilon, delta, train_epochs, current_epoch):
    processed_grads = []
    grad_norm = optax.global_norm(grad)
    # print('grad', grad_norm)
    coeff = jnp.minimum(1.0, safe_div(clip_value, grad_norm, 1e-10))
    processed_grads = jax.tree_util.tree_map(lambda x: x * coeff, grad)
    
    pruning_method, index_noise_weight, pruning_amount = pruning_key
    if pruning_method == 'None':
        noise_gen = noiseGen(epsilon, delta, train_epochs, clip_value, pruning_method, index_noise_weight)
        noise = noise_gen.generate_noise(grad, 1)
        noise_norm = optax.global_norm(noise)
        # print('noise_norm', noise_norm)
        processed_grads = jax.tree_util.tree_map(lambda x,y: x+y, noise, processed_grads)
    elif pruning_method == 'TopK':
        pruning_eps = index_noise_weight * epsilon
        noise_eps = (1 - index_noise_weight) * epsilon
        noise_gen = noiseGen(noise_eps, delta, train_epochs, clip_value, pruning_method, index_noise_weight)
        noise = noise_gen.generate_noise(grad, 1)
        noise_norm = optax.global_norm(noise)
        linear_pruning_amount = 99.9 - (99.9 - pruning_amount)*current_epoch/train_epochs
        parasum, group_num = get_netNum(grad)
        # print(type(parasum))
        # print(type(group_num))
        
        theta = pruning_eps / (parasum * jnp.where(linear_pruning_amount > 50, 100 - linear_pruning_amount, linear_pruning_amount) / 100) / group_num
        mallows_mode=mallows_model_256.MallowsModel(256, theta, pruning_amount)
        tree_value, tree_def=jax.tree_util.tree_flatten(grad)
        # print('mask_leaves', tree_mask_def)
        tree_mask_value= map(lambda x: mallows_mode.model(x, linear_pruning_amount), tree_value)
        mask=jax.tree_util.tree_unflatten(tree_def, tree_mask_value)
        # print(mask)
        # print()
        # exit(0)
        processed_grads = jax.tree_util.tree_map(lambda x,y,z: (x+y)*z, noise, processed_grads, mask)
    return processed_grads
