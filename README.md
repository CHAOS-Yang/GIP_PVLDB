# GIP code 

This codebase provides an algorithm of DPSGD with index pruning that builds off and rewrites [Jax-Privacy](https://github.com/deepmind/jax_privacy).

## Installation<a id="installation"></a>

**Note:** to ensure that your installation is compatible with your local
accelerators such as a GPU, we recommend to first follow the corresponding
instructions to install [TensorFlow](https://github.com/tensorflow/tensorflow#install)
and [JAX](https://github.com/google/jax#installation).

* Then the code can be installed so that local modifications to the code are
reflected in imports of the package:

```
cd jax_privacy
pip install -e .
```

## Reproducing Results<a id="reproducing-results"></a>

## How to Cite This Repository <a id="citing"></a>
If you use code from this repository, please cite the following reference:

```
@software{gip2023yang,
  author = {Jungang, Yang and Liyao, Xiang and Hangyu, Ye and Pengzhi, Chu and Xinbing, Wang and Chenghu, Zhou},
  title = {Improving Differentially-Private Deep Learning with Gradients Index Pruning},
  url = {https://github.com/CHAOS-Yang/GIP_PVLDB},
  version = {0.1.0},
  year = {2023},
}
```

