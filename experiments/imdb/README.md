# IMDB Experiments

Reproducing experiments in the paper.

- An experiment can be run by executing from this directory:

```
python train.py > log/log.txt
```

- You can train the model with GIP method with:

```
python train.py --pruning_method TopK_first  > log/log.txt
```

We also provide the [run.sh](run.sh) to facilitate adjustment of parameter settings.