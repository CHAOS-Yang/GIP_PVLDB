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
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def find_means_for_continuous_types(X):
    means = []
    for col in range(len(X[0])):
        summ = 0
        count = 0.000000000000000000001
        for value in X[:, col]:
            if isFloat(value): 
                summ += float(value)
                count +=1
        means.append(summ/count)
    return means

def isFloat(string):
    # credits: http://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
    try:
        float(string)
        return True
    except ValueError:
        return False

def prepare_data(raw_data, means):
    inputs = (
        ("age", ("continuous",)), 
        ("workclass", ("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked")), 
        ("fnlwgt", ("continuous",)), 
        ("education", ("Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")), 
        ("education-num", ("continuous",)), 
        ("marital-status", ("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse")), 
        ("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces")), 
        ("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")), 
        ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")), 
        ("sex", ("Female", "Male")), 
        ("capital-gain", ("continuous",)), 
        ("capital-loss", ("continuous",)), 
        ("hours-per-week", ("continuous",)), 
        ("native-country", ("United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"))
    )
    input_shape = []
    for i in inputs:
        count = len(i[1])
        input_shape.append(count)
    input_dim = sum(input_shape)
    
    X = raw_data[:, :-1]
    y = raw_data[:, -1:]
    
    # X:
    def flatten_persons_inputs_for_model(person_inputs):
        # global means
        float_inputs = []

        for i in range(len(input_shape)):
            features_of_this_type = input_shape[i]
            is_feature_continuous = features_of_this_type == 1

            if is_feature_continuous:
                mean = means[i]
                if isFloat(person_inputs[i]):
                    scale_factor = 1/(2*mean)  # we prefer inputs mainly scaled from -1 to 1. 
                    float_inputs.append(float(person_inputs[i])*scale_factor)
                else:
                    float_inputs.append(1/2)
            else:
                for j in range(features_of_this_type):
                    feature_name = inputs[i][1][j]

                    if feature_name == person_inputs[i]:
                        float_inputs.append(1.)
                    else:
                        float_inputs.append(0)
        return float_inputs
    
    new_X = []
    for person in range(len(X)):
        formatted_X = flatten_persons_inputs_for_model(X[person])
        new_X.append(formatted_X)
    new_X = np.array(new_X)
    # print(new_X.shape)
    # exit(0)
    
    # y:
    new_y = []
    for i in range(len(y)):
        if y[i] == ">50k":
            new_y.append((1, 0))
        else:  # y[i] == "<=50k":
            new_y.append((0, 1))
    new_y = np.array(new_y)
    
    return new_X, new_y

def build_train_input_dataset(
    *,
    dataset: data_info.Dataset,
    image_size_train: Tuple[int, int],
    augmult: int,
    random_crop: bool,
    random_flip: bool,
    batch_size_per_device_per_step: int,
) -> Iterator[Dict[str, chex.Array]]:
  
    training_data = np.genfromtxt('/home/jungang/dataset/adult/adult.data.txt', delimiter=', ', dtype=str, autostrip=True)
    test_data = np.genfromtxt('/home/jungang/dataset/adult/adult.test.txt', delimiter=', ', dtype=str, autostrip=True)

    means = find_means_for_continuous_types(np.concatenate((training_data, test_data), 0))
    dataset_1 = prepare_data(training_data, means)
    ds = tf.data.Dataset.from_tensor_slices(dataset_1)
    # print(ds)
    ds = ds.shuffle(buffer_size=50000)
    ds = ds.repeat()
    ds = ds.batch(
      batch_size_per_device_per_step, drop_remainder=True)
    ds = ds.map(lambda images, labels: {'images': images, 'labels': labels})
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.as_numpy_iterator()
    # x_test, y_test = prepare_data(test_data, means)
    return ds


def build_eval_input_dataset(
    *,
    dataset: data_info.Dataset,
    image_size_eval: Tuple[int, int],
    batch_size_eval: int,
) -> Iterator[Dict[str, chex.Array]]:

    training_data = np.genfromtxt('/home/jungang/dataset/adult/adult.data.txt', delimiter=', ', dtype=str, autostrip=True)
    test_data = np.genfromtxt('/home/jungang/dataset/adult/adult.test.txt', delimiter=', ', dtype=str, autostrip=True)

    means = find_means_for_continuous_types(np.concatenate((training_data, test_data), 0))
    testset = prepare_data(test_data, means)
    ds = tf.data.Dataset.from_tensor_slices(testset)
    ds = ds.batch(
      batch_size_eval, drop_remainder=True)
    ds = ds.map(lambda images, labels: {'images': images, 'labels': labels})
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.as_numpy_iterator()
    # x_test, y_test = prepare_data(test_data, means)
    return ds
