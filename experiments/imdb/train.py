import argparse
import time

import chex
import numpy
import jax
import jax.numpy as np
from typing import Any, Dict, Mapping, Optional, Tuple, Callable
import optax

import tqdm
from tqdm.contrib import tmap


from jax import jit, grad, random, nn
from jax.example_libraries import optimizers, stax

import sys
# sys.path.append('/home/lxiang_stu6/jungang/gip_branch')
from jax_privacy.src.training import grad_clipping
import mallows_model_256



import functools 

from utils import load_data
from models import Embedding, LSTM
from perturb import perturb_grad, noiseGen



Aux = chex.ArrayTree
Params = chex.ArrayTree
GradParams = Params
PruningFn = Callable[[GradParams], Tuple[GradParams, Aux]]

@jit
def loss_fun_batch(params, batch, network_state, rng):
    """
    The idxes of the batch indicate which nodes are used to compute the loss.
    """
    inputs, targets = batch["inputs"], batch["labels"]
    preds = predict_fun(params, inputs, rng=rng)
    ce_loss = -np.mean(np.sum(preds * targets, axis=1))
    # l2_loss = 5e-4 * optimizers.l2_norm(params)**2 # tf doesn't use sqrt
    # print('celoss', ce_loss)
    # print('l2_loss', l2_loss)
    return ce_loss, (network_state)

@jit
def loss_fun(params, inputs, network_state, rng):
    """
    The idxes of the batch indicate which nodes are used to compute the loss.
    """
    input, target = inputs["inputs"], inputs["labels"]
    # print(target)
    preds = predict_fun(params, input, rng=rng)
    # print(np.sum(preds * target, axis=1).shape)
    ce_loss = -np.sum(preds * target, axis=1)
    # ce_loss = np.mean(optax.softmax_cross_entropy(preds, target))
    # l2_loss = 5e-4 * optimizers.l2_norm(params)**2 # tf doesn't use sqrt
    return np.mean(ce_loss), (network_state, None, ce_loss)

@jit
def l2_loss(params, batch):
    """
    The idxes of the batch indicate which nodes are used to compute the loss.
    """
    # preds = predict_fun(params, inputs, adj, is_training=is_training, rng=rng)
    # print(params)
    # ce_loss = -np.mean(np.sum(preds[np.array(idx)] * targets[np.array(idx)], axis=1))
    l2_loss = 5e-4 * optimizers.l2_norm(params)**2 # tf doesn't use sqrt
    # print('celoss', ce_loss)
    # print('l2_loss', l2_loss)
    return l2_loss   


@jit
def accuracy(params, batch, network_state, rng):

    inputs, targets = batch["inputs"], batch["labels"]
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict_fun(params, inputs, rng=rng), axis=1)

    return np.mean(predicted_class == target_class)





@jit
def loss_accuracy(params, batch, network_state, rng):

    inputs, targets = batch["inputs"], batch["labels"]
    preds = predict_fun(params, inputs, rng=rng)
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(preds, axis=1)

    loss = -np.mean(np.sum(preds * targets, axis=1))
    acc = np.mean(predicted_class == target_class)
    return loss, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--pruning_method', type=str, default='None')
    parser.add_argument('--pruning_amount', type=float, default=10)
    parser.add_argument('--idx_noise_weight', type=float, default=0.01)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='imdb')
    parser.add_argument('--eps', type=float, default=4)
    parser.add_argument('--delta', type=float, default=0.0001)
    parser.add_argument('--maxlen', type=int, default=80, help="length of words in an input")
    parser.add_argument('--num_words', type=int, default=10000, help="vocabulary size")
    parser.add_argument("--glove_dir", type=str, default="./data/glove6B")
    parser.add_argument("--batch_size", type=int, default = 4096)
    parser.add_argument("--clip_value", type=float, default=1)
    args = parser.parse_args()

    if args.dataset != 'imdb':
        print("Wrong dataset. This code is only for IMDB dataset")

    # Load data
    print("\nStarting loading")
    train_set, val_set, test_set, embedding_matrix = load_data(args.dataset, args.glove_dir, args.num_words, args.maxlen)
    (train_x , train_y) = train_set
    (val_x, val_y) = val_set
    (test_x, test_y) = test_set

    
    train_all = {"inputs": train_x, "labels": train_y}
    val_batch = {"inputs": val_x, "labels": val_y}

    print("Loaded data info:")
    print("    train_x: {}, train_y: {}".format(train_x.shape, train_y.shape))
    print("    val_x: {}, val_y: {}".format(val_x.shape, val_y.shape))
    print("    test_x: {}, test_y: {}".format(test_x.shape, test_y.shape))

    clip_value = args.clip_value
    rng_key = random.PRNGKey(args.seed)
    dropout = args.dropout
    step_size = args.lr
    num_epochs = args.epochs
    early_stopping = args.early_stop
    num_training = train_x.shape[0]

    # clip_value = args.clip_value
    clipping_norm = args.clip_value
    rescale_to_unit_norm = False
    pruning_method = args.pruning_method
    pruning_amount = args.pruning_amount
    index_noise_weight = args.idx_noise_weight
    pruning_key = (pruning_method, index_noise_weight, pruning_amount)
    epsilon = args.eps
    delta = args.delta
    batch_size = args.batch_size
    max_step = int(num_epochs * num_training / batch_size)


    noise_gen = noiseGen(epsilon, delta, max_step, clip_value, pruning_method, index_noise_weight, batch_size, num_training)
    value_and_clipped_grad = functools.partial(
            grad_clipping.value_and_clipped_grad_vectorized,
            clipping_fn=grad_clipping.global_clipping(
                clipping_norm=clipping_norm,
                rescale_to_unit_norm=rescale_to_unit_norm
            )
    )


    init_fun, predict_fun = stax.serial(
        stax.Dense(32),              # 100 * 32 + 32
        stax.elementwise(nn.relu),
        LSTM(32),                    # ((32 + 32) * 32 + 32) * 4 + 32 * 1 + 1 # ((embedding_size + hidden_size) * hidden_size + hidden_size) * 4
        stax.elementwise(nn.relu),
        stax.Flatten,
        stax.Dense(16),              # (80 * 32) * 16 + 16
        stax.elementwise(nn.relu),
        stax.Dense(2),               # 16 * 2 + 2
        stax.LogSoftmax
    )

    input_shape = (-1, args.maxlen, 100)
    rng_key, init_key = random.split(rng_key)

    _, init_params = init_fun(init_key, input_shape)

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    
    paramsNum = None

    def batch_pruning_topk_split(
        batch_pruning_amount: chex.Array,
        mallows_mode,
    ) -> PruningFn:

        def pruning_fn(grad_mask: GradParams) -> Tuple[GradParams, Aux]:
            # tree_value, tree_def=jax.tree_util.tree_flatten(grad)
            tree_mask_value, tree_mask_def=jax.tree_util.tree_flatten(grad_mask)
            # print('mask_leaves', tree_mask_def)
            tree_mask_value= map(leaves_split, tree_mask_value)
            tree_output=jax.tree_util.tree_unflatten(tree_mask_def, tree_mask_value)
            # del tree_value, tree_mask_value, tree_def, tree_mask_def
            # return jax.tree_util.tree_map(lambda x, y: x*y, tree_output, grad), jax.tree_util.tree_map(lambda x, y: x*y, tree_output, grad_mask)
            return tree_output

        def leaves_split(mask_leaves):
            # print(mask_leaves.shape)
            mask = mask_leaves.reshape(-1)
            mask = mallows_mode.model(mask, batch_pruning_amount)
            return mask.reshape(mask_leaves.shape)

        return pruning_fn

    def batch_pruning_random(  
        batch_pruning_amount: chex.Array,
    ) -> PruningFn:

        def pruning_fn(grad: GradParams) -> Tuple[GradParams, Aux]:
            tree_value, tree_def=jax.tree_util.tree_flatten(grad)
            tree_value=map(leaves_pruning,tree_value)
            tree_output=jax.tree_util.tree_unflatten(tree_def,tree_value)
            return tree_output

        def leaves_pruning(leaves):
            random_grad = jax.random.normal(rng_key,np.shape(leaves))
            quantile=np.percentile(np.abs(random_grad), 100-batch_pruning_amount)
            # return jax.tree_util.tree_map(lambda x: x * get_mask(random_grad, quantile), leaves)
            return np.where(np.abs(random_grad)> quantile, 1, 0)
    
        def get_mask(x, quantile):
            return np.where(np.abs(x)> quantile, 1, 0)

        return pruning_fn


    @jit
    def get_sum(mask):
        leaves_value, structure = jax.tree_util.tree_flatten(mask)
        sum = 0
        for leave in leaves_value:
          sum += np.sum(leave)
        return sum
    
    def get_netNum(params):
        leaves_value, structure = jax.tree_util.tree_flatten(params)
        group_num = 0
        num = 0
        for leave in leaves_value:
            num += np.size(leave)
            # print(leave.shape)
            group_num += np.ceil(np.size(leave) / 256.)
        return num, group_num
    
    @jit
    def update(i, opt_state, batch, paramsNum, global_step):
        params = get_params(opt_state)
        
        # grads = grad(loss_fun)(params, batch)
        pruning_eps_step = 0.01 * epsilon
        
        if pruning_method == 'TopK_first':

            avg_clean_grads, unused_aux = jax.grad(loss_fun_batch, has_aux=True)(
                params, batch, None, rng_key)

            linear_pruning_amount = 99.9 - (99.9 - pruning_amount)*global_step/max_step
            # print("************************", linear_pruning_amount)
            theta = pruning_eps_step / (paramsNum  * 2 * np.where(linear_pruning_amount > 50, 100 - linear_pruning_amount, linear_pruning_amount) / 100)
            # theta = 0.1
            mallows_mode=mallows_model_256.MallowsModel(256, theta, linear_pruning_amount)
            batch_pruningFn = batch_pruning_topk_split(linear_pruning_amount, mallows_mode=mallows_mode)

            mask= batch_pruningFn(avg_clean_grads)
            # mask = None
            # mask = jax.tree_util.tree_map(lambda x: np.where(x == 0, 0, 1), grad_pruned)
            mask_norm = get_sum(mask)
            # mask_norm = 0
        elif pruning_method == 'Random':

            avg_clean_grads, unused_aux = jax.grad(loss_fun_batch, has_aux=True)(
                params, batch, None, rng_key)
            batch_pruningFn = batch_pruning_random(99.9 - np.exp(np.log(99.9 - pruning_amount)*global_step/max_step))   
            mask=batch_pruningFn(avg_clean_grads)
            mask_norm = get_sum(mask)
            # cos = cos_sim.cos_sim(grads, avg_grads)
            print('pruning finished')
        else:
            # loss_fun = functools.partial(loss_fun, frozen_params=frozen_params)
            mask = None

        if mask is None:
            mask_norm = 0
        else:
            mask_norm = get_sum(mask)

        (loss, (network_state)), (grads, grad_norm, origin_grads) = value_and_clipped_grad(loss_fun)(
                params, batch, None, rng_key, mask)
        # exit(0)
        
        # grads_norm = optax.global_norm(origin_grads)
        # avg_norm = optax.global_norm(avg_clean_grads)
        # delta_grad = jax.tree_util.tree_map(lambda x,y: np.abs(x-y), origin_grads, avg_clean_grads)
        # print('distance', optax.global_norm(delta_grad))
        # num,_ = get_netNum(grads)
        # print(num)
        # print('grad', grad_norm)
        noisey_grad = perturb_grad(grads, pruning_key, noise_gen, rng_key, batch_size, mask)
        # summed_grad = jax.tree_util.tree_map(lambda x,y: x+y, noisey_grad, grad(l2_loss)(params, batch))
        summed_grad = noisey_grad
        return opt_update(global_step, summed_grad, opt_state) , (np.median(grad_norm), mask_norm)

    opt_state = opt_init(init_params)

    # train
    print("\nStarting training...")

    val_values = []
    paramsNum = None
    for epoch in range(num_epochs):
        start_time = time.time()

        
        if paramsNum is None:
            params = get_params(opt_state)
            paramsNum, group_num = get_netNum(params)
            print('\n==> Number of model:', paramsNum)
        grad_norm_list = []
        mask_norm_list = []
        for b in tqdm.trange(train_x.shape[0] // batch_size):
            train_batch = {"inputs": train_x[b * batch_size : (b + 1) * batch_size], "labels": train_y[b * batch_size : (b + 1) * batch_size]}
            global_step = int(epoch * 25000 / batch_size) + b
            opt_state, (grad_norm, mask_norm) = update(epoch, opt_state, train_batch, paramsNum, global_step)
            grad_norm_list.append(grad_norm.tolist())
            mask_norm_list.append(mask_norm)

        epoch_time = time.time() - start_time

        params = get_params(opt_state)

        train_loss, train_acc = loss_accuracy(params, train_all, None, rng_key)
        val_loss, val_acc = loss_accuracy(params, val_batch, None, rng_key)
        val_values.append(val_loss.item())
        print(f"Iter {epoch}/{num_epochs} ({epoch_time:.4f} s) train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        print('grad_norm', grad_norm_list)
        # print('mask_norm', mask_norm_list)
        # new random key at each iteration, othwerwise dropout uses always the same mask 
        rng_key, _ = random.split(rng_key)
        if epoch > early_stopping and val_values[-1] > numpy.mean(val_values[-(early_stopping+1):-1]):
            print("Early stopping...")
            break


    # test
    test_batch = {"inputs": test_x, "labels": test_y}
    test_acc = accuracy(params, test_batch, None, rng_key)
    print(f'Test set acc: {test_acc}')

