# coding=utf-8
# Copyright 2022 Jungang Yang Limited.
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

import jax.numpy as np
from jax import lax, random
from jax.example_libraries import stax
from jax.example_libraries.stax import Relu, LogSoftmax
from jax.nn.initializers import glorot_normal, glorot_uniform, normal, uniform, zeros

import jax.nn as nn
import haiku as hk
import numpy as onp




def Dropout(rate):
    """
    Layer construction function for a dropout layer with given rate.
    This Dropout layer is modified from stax.experimental.Dropout, to use
    `is_training` as an argument to apply_fun, instead of defining it at
    definition time.

    Arguments:
        rate (float): Probability of keeping and element.
    """
    def init_fun(rng, input_shape):
        return input_shape, ()
    def apply_fun(params, inputs, is_training, **kwargs):
        rng = kwargs.get('rng', None)
        if rng is None:
            msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
                   "argument. That is, instead of `apply_fun(params, inputs)`, call "
                   "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                   "jax.random.PRNGKey value.")
            raise ValueError(msg)
        keep = random.bernoulli(rng, rate, inputs.shape)
        outs = np.where(keep, inputs / rate, 0)
        # if not training, just return inputs and discard any computation done
        out = lax.cond(is_training, outs, lambda x: x, inputs, lambda x: x)
        return out
    return init_fun, apply_fun


def Embedding(embedding_matrix):
    embedding_matrix = embedding_matrix
    def init_fun(rng, input_shape):
        params = []

        return input_shape, params

    def apply_fun(params, x, is_training=False, **kwargs):
        if embedding_matrix is not None:
            tmp = onp.reshape(x, (x.shape[0],-1))
            x = onp.reshape(embedding_matrix[tmp], x.shape + (embedding_matrix.shape[1],)) 
            x = embedding_matrix[x.reshape(-1)].reshape(x.shape + (embedding_matrix.shape[1],))
        else:
            print("Embedding matrix not found")
            raise
        return x

    return init_fun, apply_fun

def LSTM_haiku(hidden):
    def unroll_net(seqs: np.ndarray):
        """Unrolls an LSTM over seqs, mapping each output to a scalar."""
        # seqs is [B, T, F].
        core = hk.LSTM(hidden)
        batch_size = seqs.shape[0]
        outs, state = hk.dynamic_unroll(core, seqs, core.initial_state(batch_size), time_major=False)
        # We could include this Linear as part of the recurrent core!
        # However, it's more efficient on modern accelerators to run the linear once
        # over the entire sequence than once per sequence element.
        return outs, state

    return unroll_net

def LSTM(hidden):
    model = hk.transform(LSTM_haiku(hidden))
    def init_fun(rng, input_shape):
        
        inputs = np.zeros((1, ) + input_shape[1:]) # construct an inputs for lstm
        params = model.init(rng, inputs)
        output_shape = input_shape[:-1] + (hidden, )
        return output_shape, params

    def apply_fun(params, x, is_training=False, **kwargs):
        x, _ = model.apply(params, None, x)
        return x

    return init_fun, apply_fun