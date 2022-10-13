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

"""Two-layer CNN for MNIST (mainly for membership inference attacks)."""
import functools

import chex
import haiku as hk
import jax

from jax_privacy.src.training.image_classification.models import common


class MLP(hk.Module):
  """Hard-coded two-layer CNN."""

  def __init__(
      self,
      num_classes: int = 10,
      activation: common.Activation = jax.nn.relu
  ):
    super().__init__()

    # First linear layer.
    self._linear_1 = hk.Linear(64, name='linear_1')

    # self._linear_2 = hk.Linear(32, name='linear_2')

    # Classification layer.
    self._logits_module = hk.Linear(num_classes, name='linear_2')

    self._activation = activation

  def __call__(self, inputs: chex.Array, is_training: bool) -> chex.Array:
    return hk.Sequential([
        hk.Flatten(),
        self._linear_1,
        self._activation,
        self._logits_module,
    ])(inputs)
