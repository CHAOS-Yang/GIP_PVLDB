from jax_privacy.src.training.image_classification import config_base
from jax_privacy.src.training.image_classification import data
from ml_collections import config_dict as configdict


@config_base.wrap_get_config
def get_config(config):
  """Experiment config."""
  config.checkpoint_dir = '../../result/eps=1/random10'  
  config.experiment_kwargs = configdict.ConfigDict(
      dict(
          config=dict(
              num_updates=875,  ##
              checkpoint_dir = '../../result/eps=1/random10',
              save_final_checkpoint_as_npy = True,
              load_checkpoint_from_npy = False,
              load_checkpoint_dir = None,
              optimizer=dict(
                  name='sgd',
                  lr=dict(
                      init_value=2,
                      decay_schedule_name=None,
                      decay_schedule_kwargs=None,
                      relative_schedule_kwargs=None,
                      # decay_schedule_name='cosine_decay_schedule',
                      # decay_schedule_kwargs=configdict.ConfigDict(
                      #     {
                      #         'init_value': 1.0,
                      #         'decay_steps': 1.0,
                      #         'alpha': 0.0,
                      #     },
                      #     convert_dict=False),
                      # relative_schedule_kwargs=['decay_steps'],
                      ),
                  kwargs=dict(),
              ),
              model=dict(
                  model_type='wideresnet',
                  model_kwargs=dict(
                      depth=16,
                      width=4,
                  ),
                  restore=dict(
                      path=None,
                      params_key=None,
                      network_state_key=None,
                      layer_to_reset=None,
                  ),
              ),
              training=dict(
                  batch_size=dict(
                      init_value=4096,
                      per_device_per_step=128,   #4*64 per step 
                      scale_schedule=None,  # example: {'2000': 8, '4000': 16},
                  ),
                  weight_decay=0.0,  # L-2 regularization,
                  train_only_layer=None,
                  dp=dict(
                      target_delta=1e-5,
                      clipping_norm=1.0,  # float('inf') or None to deactivate
                      stop_training_at_epsilon=1.0,  # None,
                      rescale_to_unit_norm=False,
                      noise=dict(
                          std_relative=10,  # noise multiplier
                          ),
                      # Set the following flag to auto-tune one of:
                      # * 'batch_size'
                      # * 'std_relative'
                      # * 'stop_training_at_epsilon'
                      # * 'num_updates'
                      # Set to `None` to deactivate auto-tunning
                      auto_tune=None,  # 'num_updates',  # None,
                      per_example_pruning_amount=30,  ##(%)   cant be 100
                      batch_pruning_amount=10,  ##(#)
                      datalens_pruning=False,
                      batch_pruning_method="Random",
                      index_noise_weight=0,
                      ),
                  logging=dict(
                      grad_clipping=True,
                      grad_alignment=False,
                      snr_global=True,  # signal-to-noise ratio across layers
                      snr_per_layer=False,  # signal-to-noise ratio per layer
                  ),
              ),
              averaging=dict(
                  ema=dict(
                      coefficient=0.9999,
                      start_step=0,
                  ),
                  polyak=dict(
                      start_step=0,
                  ),
              ),
              
              data=dict(
                  dataset=data.get_dataset(
                      name='cifar10',
                      train_split='train_valid',  # 'train' or 'train_valid'
                      eval_split='test',  # 'valid' or 'test'
                  ),
                  random_flip=True,
                  random_crop=True,
                  augmult=16,  # implements arxiv.org/abs/2105.13343
                  ),
              evaluation=dict(
                  batch_size=100,
              ))))

  return config
