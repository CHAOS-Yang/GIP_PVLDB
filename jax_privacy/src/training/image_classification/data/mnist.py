import numpy as np
import chex
import jax.numpy as jnp
from jax_privacy.src.training.image_classification.data import data_info
from torch.utils import data
from torchvision.datasets import MNIST
from typing import Dict, Iterator, Optional, Tuple



def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

def build_train_input_dataset(
    *,
    dataset: data_info.Dataset,
    image_size_train: Tuple[int, int],
    augmult: int,
    random_crop: bool,
    random_flip: bool,
    batch_size_per_device_per_step: int,
) -> Iterator[Dict[str, chex.Array]]:
  # batch_size = 4096
  # Define our dataset, using torch datasets
  mnist_dataset = MNIST('/home/jungang/dataset/mnist/MNIST/', download=True, transform=FlattenAndCast())
  training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size_per_device_per_step, num_workers=0)
  for x, y in training_generator:
      y = one_hot(y, 10)
      # print(x.shape)
  return training_generator

def build_eval_input_dataset(
    *,
    dataset: data_info.Dataset,
    image_size_eval: Tuple[int, int],
    batch_size_eval: int,
) -> Iterator[Dict[str, chex.Array]]:
  mnist_dataset = MNIST('/home/jungang/dataset/mnist/MNIST/', download=True, transform=FlattenAndCast(),train=False)
  testing_generator = NumpyLoader(mnist_dataset, batch_size=batch_size_eval, num_workers=0)
  for x, y in testing_generator:
      y = one_hot(y, 10)
  return testing_generator

build_eval_input_dataset(dataset='mnist', image_size_eval=(28,28), batch_size_eval=64)