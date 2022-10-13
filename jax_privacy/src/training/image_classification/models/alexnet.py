import functools

import chex
import haiku as hk
import haiku.initializers as hk_init
from jax_privacy.src.training.image_classification.models import common
import jax

#from jax_privacy.src.training.image_classification.models import common


class AlexNet(hk.Module):

  def __init__(
      self,
      num_classes: int = 10,
      activation = jax.nn.relu
  ):
    super().__init__()

    # All conv layers have a kernel shape of 3 and a stride of 1.
    self._conv_1 = common.WSConv2D(
        output_channels=64,
        kernel_shape=3,
        name='conv2d_1',
    )
    self._conv_2 = common.WSConv2D(
        output_channels=192,
        kernel_shape=3,
        name='conv2d_2',
    )
    self._conv_3 = common.WSConv2D(
        output_channels=384,
        kernel_shape=3,
        name='conv2d_3',
    )
    self._conv_4 = common.WSConv2D(
        output_channels=256,
        kernel_shape=3,
        name='conv2d_4',
    )
    self._conv_5 = common.WSConv2D(
        output_channels=256,
        kernel_shape=3,
        name='conv2d_5',
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

    # Classification layer.
    self._logits_module = hk.Linear(num_classes, name='output')
    self._pool = functools.partial(
        hk.max_pool,
        window_shape=[1,2,2,1],
        strides=[1,2,2,1],
        padding='SAME',
    )
    self._dropout = functools.partial(
        hk.dropout,
        hk.next_rng_key(),
        0.5,
    )
    self._activation = activation

  def __call__(self, inputs: chex.Array, is_training: bool) -> chex.Array:
    return hk.Sequential([
        self._conv_1,
        self._activation,
        self._gn_1,
        self._pool,
        self._conv_2,
        self._activation,
        self._gn_2,
        self._pool,
        self._conv_3,
        self._activation,
        self._gn_3,
        self._conv_4,
        self._activation,
        self._gn_4,
        self._conv_5,
        self._activation,
        self._gn_5,
        self._pool,
        hk.Flatten(),
        self._linear_1,
        self._activation,
        self._gn_6,
        self._dropout,
        self._linear_2,
        self._activation,
        self._gn_7,
        self._dropout,
        self._logits_module,
    ])(inputs)