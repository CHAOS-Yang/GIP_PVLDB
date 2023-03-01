# coding=utf-8
# Copyright 2022 DeepMind Technologies and Jungang Yang Limited.
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

import functools

import chex
import haiku as hk
import haiku.initializers as hk_init
import jax

#from jax_privacy.src.training.image_classification.models import common


class VGG16(hk.Module):

  def __init__(
      self,
      num_classes: int = 10,
      activation = jax.nn.relu
  ):
    super().__init__()

    # All conv layers have a kernel shape of 3 and a stride of 1.
    self._conv_1 = hk.Conv2D(
        output_channels=64,
        kernel_shape=3,
        name='conv2d_1',
    )
    self._conv_2 = hk.Conv2D(
        output_channels=64,
        kernel_shape=3,
        name='conv2d_2',
    )
    self._conv_3 = hk.Conv2D(
        output_channels=128,
        kernel_shape=3,
        name='conv2d_3',
    )
    self._conv_4 = hk.Conv2D(
        output_channels=128,
        kernel_shape=3,
        name='conv2d_4',
    )
    self._conv_5 = hk.Conv2D(
        output_channels=256,
        kernel_shape=3,
        name='conv2d_5',
    )
    self._conv_6 = hk.Conv2D(
        output_channels=256,
        kernel_shape=3,
        name='conv2d_6',
    )
    self._conv_7 = hk.Conv2D(
        output_channels=256,
        kernel_shape=3,
        name='conv2d_7',
    )
    self._conv_8 = hk.Conv2D(
        output_channels=512,
        kernel_shape=3,
        name='conv2d_8',
    )
    self._conv_9 = hk.Conv2D(
        output_channels=512,
        kernel_shape=3,
        name='conv2d_9',
    )
    self._conv_10 = hk.Conv2D(
        output_channels=512,
        kernel_shape=3,
        name='conv2d_10',
    )
    self._conv_11 = hk.Conv2D(
        output_channels=512,
        kernel_shape=3,
        name='conv2d_11',
    )
    self._conv_12 = hk.Conv2D(
        output_channels=512,
        kernel_shape=3,
        name='conv2d_12',
    )
    self._conv_13 = hk.Conv2D(
        output_channels=512,
        kernel_shape=3,
        name='conv2d_13',
    )

    # First linear layer.
    self._linear_1 = hk.Linear(512, name='linear_1')

    self._linear_2 = hk.Linear(512, name='linear_2')

    self._gn_1=hk.GroupNorm(32,name='gn_1')
    self._gn_2=hk.GroupNorm(32,name='gn_2')
    self._gn_3=hk.GroupNorm(32,name='gn_3')
    self._gn_4=hk.GroupNorm(32,name='gn_4')
    self._gn_5=hk.GroupNorm(32,name='gn_5')
    self._gn_6=hk.GroupNorm(32,name='gn_6')
    self._gn_7=hk.GroupNorm(32,name='gn_7')
    self._gn_8=hk.GroupNorm(32,name='gn_8')
    self._gn_9=hk.GroupNorm(32,name='gn_9')
    self._gn_10=hk.GroupNorm(32,name='gn_10')
    self._gn_11=hk.GroupNorm(32,name='gn_11')
    self._gn_12=hk.GroupNorm(32,name='gn_12')
    self._gn_13=hk.GroupNorm(32,name='gn_13')

    # Classification layer.
    self._logits_module = hk.Linear(num_classes, name='output')
    self._pool = functools.partial(
        hk.max_pool,
        window_shape=[1,2,2,1],
        strides=[1,2,2,1],
        padding='VALID',
    )

    self._activation = activation

  def __call__(self, inputs: chex.Array, is_training: bool) -> chex.Array:
    return hk.Sequential([
        self._conv_1,
        self._gn_1,
        self._activation,
        self._conv_2,
        self._gn_2,
        self._activation,
        self._pool,
        self._conv_3,
        self._gn_3,
        self._activation,
        self._conv_4,
        self._gn_4,
        self._activation,
        self._pool,
        self._conv_5,
        self._gn_5,
        self._activation,
        self._conv_6,
        self._gn_6,
        self._activation,
        self._conv_7,
        self._gn_7,
        self._activation,
        self._pool,
        self._conv_8,
        self._gn_8,
        self._activation,
        self._conv_9,
        self._gn_9,
        self._activation,
        self._conv_10,
        self._gn_10,
        self._activation,
        self._pool,
        self._conv_11,
        self._gn_11,
        self._activation,
        self._conv_12,
        self._gn_12,
        self._activation,
        self._conv_13,
        self._gn_13,
        self._activation,
        self._pool,
        hk.Flatten(),
        self._linear_1,
        self._activation,
        self._linear_2,
        self._activation,
        self._logits_module,
    ])(inputs)
    '''
    x=self._conv_1(inputs)
    x=self._activation(x)
    print('shape1',x.shape)
    x=self._conv_2(x)
    x=self._activation(x)
    x=self._pool(x)
    print('shape2',x.shape)
    x=self._conv_3(x)
    x=self._activation(x)
    x=self._conv_4(x)
    x=self._activation(x)
    x=self._pool(x)
    x=self._conv_5(x)
    x=self._activation(x)
    x=self._conv_6(x)
    x=self._activation(x)
    x=self._conv_7(x)
    x=self._activation(x)
    x=self._pool(x)
    x=self._conv_8(x)
    x=self._activation(x)
    x=self._conv_9(x)
    x=self._activation(x)
    x=self._conv_10(x)
    x=self._activation(x)
    x=self._pool(x)
    x=self._conv_11(x)
    x=self._activation(x)
    x=self._conv_12(x)
    x=self._activation(x)
    x=self._conv_13(x)
    x=self._activation(x)
    x=self._pool(x)
    print('shape3',x.shape)
    x = hk.Flatten()(x)
    print('shape4',x.shape)
    x=self._linear_1(x)
    x=self._activation(x)
    print('shape5',x.shape)
    x=self._linear_2(x)
    x=self._activation(x)
    x=self._logits_module(x)
    print('shape6',x.shape)
    return x
    '''