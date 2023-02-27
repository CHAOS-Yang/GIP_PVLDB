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
_total_steps = 800
_pruning_method = None
_index_noise_weight = 0.01
_batch_size = 4096

class GauMechanism(object):
    def __init__(self, num_training, epsilon=_epsilon, delta=_delta,
                 l2_clip_value=_l2_clip_value,
                 total_steps=_total_steps,
                 pruning_method=_pruning_method,
                 index_noise_weight=_index_noise_weight,
                 batch_size=_batch_size,
                 ):
        assert epsilon > 0, "Epsilon should be larger than 0."
        assert delta > 0, "Delta should be larger than 0."
        self._epsilon = epsilon
        self._delta = delta
        self._sample_rate_q = 1
        self._noise_iter = total_steps
        self._clip_value = l2_clip_value
        self.batch_pruning_method = pruning_method
        self._sigma = 10
        self._num_training = num_training
        self.index_noise_weight = index_noise_weight
        self.batch_size = batch_size


        self.batching = batching.VirtualBatching(
        batch_size_init=self.batch_size,
        batch_size_per_device_per_step=self.batch_size,
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
        if "TopK" in self.batch_pruning_method:
            pruning_eps = self._epsilon * self.index_noise_weight
            # pruning_eps_step = pruning_eps / self._max_num_updates / self.config.training.batch_size.init_value * self.num_training_samples
            self.accountant._dp_epsilon=self._epsilon * (1 - self.index_noise_weight)
            self._sigma=self.accountant.compute_target_sigma(self._noise_iter)
            
        else:
            pruning_eps_step = None
            self._sigma=self.accountant.compute_target_sigma(self._noise_iter)
            # sigma=self.config.training.dp.noise.std_relative
        print(self._noise_iter)
        print('noise_std=', self._sigma)

    def current_eps(self, num_updates):
        return self.accountant.compute_current_epsilon(int(num_updates))


    def generate_noise(self, grad, num_supports, rng_key):
        noiselist = []
        for l in range(len(grad)//num_supports):
            # print("layer {}: grad type: {}".format(l, type(grad[l * num_supports])))

            if type(grad[l * num_supports]) == tuple: 
                
                if len(grad[l * num_supports]) == 0: 
                    '''
                    element-wise layer: Relu, Flatten, etc
                    '''
                    noiselist.append(())
                    continue

                '''
                Dense 
                '''

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
            elif type(grad[l * num_supports]) == dict: 
                
                noise_dict = {}
                for k,v in grad[l * num_supports].items():
                    if k == "lstm/linear":
                        '''
                        LSTM
                        '''
                        noise = {}
                        for param_k, param_v in v.items():
                            l2_sen = self._clip_value
                            noise[param_k] = jax.random.normal(rng_key, param_v.shape) * l2_sen * self._sigma
                    else:
                        raise("layer {}: the grad type is dict, but key not supported! key: {}".format(l, k))
                    noise_dict[k] = noise
                
                noiselist.append(noise_dict)
            else:
                raise("Grad Type Not Supported! Type: {}".format(type(grad[l * num_supports])))

        # print(noiselist)
        return noiselist


def noiseGen(epsilon, delta, total_steps, clip_value, pruning_method, index_noise_weight, batch_size, num_training):
    noiseGen = GauMechanism(num_training=num_training, epsilon=epsilon, delta=delta, total_steps=total_steps, l2_clip_value=clip_value,pruning_method=pruning_method, index_noise_weight=index_noise_weight, batch_size=batch_size)
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

def perturb_grad(grad, pruning_key, noise_gen, rng, batch_size, mask):
    # processed_grads = []
    grad_norm = optax.global_norm(grad)
    print('grad', grad_norm)
    # coeff = 1. # jnp.minimum(1.0, safe_div(clip_value, grad_norm, 1e-10))
    # processed_grads = jax.tree_util.tree_map(lambda x: x * coeff, grad)
    processed_grads = grad 
    
    pruning_method, index_noise_weight, pruning_amount = pruning_key
    if pruning_method == 'None':
        noise = noise_gen.generate_noise(grad, 1, rng_key=rng)
        noise_norm = optax.global_norm(noise)
        print('noise_norm', noise_norm)
        processed_grads = jax.tree_util.tree_map(lambda x,y: x / batch_size + y, noise, processed_grads)
    elif pruning_method == 'TopK_first' or pruning_method == 'Random':
        noise = noise_gen.generate_noise(grad, 1, rng_key=rng)
        processed_grads = jax.tree_util.tree_map(lambda x,y,z: (x / batch_size+y)*z, noise, processed_grads, mask)
    return processed_grads
