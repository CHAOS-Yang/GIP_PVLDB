# Image Classification Experiments

Reproducing experiments in the paper.

## Intro


- An experiment can be run by executing from this directory:

```
python run_experiment.py --config=<relative/path/to/config.py> --jaxline_mode=train_eval_multithreaded
```

where the config file contains all relevant hyper-parameters for the experiment.

- The main config hyper-parameters to update per experiment are:

  - Augmult: `config.experiment_kwargs.config.data.augmult`
  - Batch-size: `config.experiment_kwargs.config.training.batch_size.init_value`
  - Learning-rate value: `config.experiment_kwargs.config.optimizer.lr.init_value`
  - Model definition: `config.experiment_kwargs.config.model`
  - Noise multiplier sigma: `config.experiment_kwargs.config.training.dp.noise.std_relative`
  - Number of updates: `config.experiment_kwargs.config.num_updates`
  - Privacy budget (delta): `config.experiment_kwargs.config.dp.target_delta`
  - Privacy budget (epsilon): `config.experiment_kwargs.config.dp.stop_training_at_epsilon`
  - Pruning Method: `config.experiment_kwargs.config.dp.batch_pruning_method`('TopK_first','Random', 'None')
  - Retain amount: `config.experiment_kwargs.config.dp.batch_pruning_amount`(0-100)

Note: we provide examples of configurations for various experiments. 

## Training from Scratch on CIFAR-10

```
python run_experiment.py --config=configs/cifar10_wrn_16_4_eps1.py --jaxline_mode=train_eval_multithreaded
```

## Fine-tuning on CIFAR

```
python run_experiment.py --config=configs/cifar10_wrn_28_10_eps1_finetune.py --jaxline_mode=train_eval_multithreaded
```

See `jax_privacy/src/training/image_classsification/config_base.py` for the available pre-trained models.

