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

"""Data loading functions for MNIST / CIFAR / SVHN."""

import functools
from typing import Dict, Iterator, Optional, Tuple

import chex
import numpy as np
import jax.numpy as jnp
from jax_privacy.src.training.image_classification.data import data_info
from jax_privacy.src.training.image_classification.data import image_dataset_loader
from jax_privacy.src.training.image_classification.data import utils
import tensorflow as tf
import tensorflow_datasets as tfds


def build_train_input_dataset(
    *,
    dataset: data_info.Dataset,
) -> Dict[str, chex.Array]:
    adj, features, labels, idx_train, idx_val, idx_test = utils.load_data(dataset, sparse=False)
    print('train_features',features.shape)
    return {
        'adj': adj,
        'features': features,
        'labels': labels,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
    }


def build_eval_input_dataset(
    *,
    dataset: data_info.Dataset,
) -> Dict[str, chex.Array]:
    adj, features, labels, idx_train, idx_val, idx_test = utils.load_data(dataset, sparse=False)
    
    # x_test, y_test = prepare_data(test_data, means)
    return {
        'adj': adj,
        'features': features,
        'labels': labels,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
    }

# train_set = build_train_input_dataset(dataset='cora')
# print(train_set['features'].shape)
# print(train_set['labels'][1])
# print(train_set['adj'].shape)
# print(len(train_set['idx_train']))
# print(len(train_set['idx_val']))
# print(len(train_set['idx_test']))
# (2708, 1433)
# [0 0 0 0 1 0 0]
# (2708, 2708)
# 140
# 500
# 1000