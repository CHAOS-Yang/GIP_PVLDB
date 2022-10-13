import argparse
import time

import numpy
import jax
import jax.numpy as np
from jax import jit, grad, random
from jax.experimental import optimizers

from utils import load_data
from models import GCN
from perturb import perturb_grad

@jit
def loss(params, batch):
    """
    The idxes of the batch indicate which nodes are used to compute the loss.
    """
    inputs, targets, adj, is_training, rng, idx = batch
    preds = predict_fun(params, inputs, adj, is_training=is_training, rng=rng)
    # print(params)
    ce_loss = -np.mean(np.sum(preds[np.array(idx)] * targets[np.array(idx)], axis=1))
    # l2_loss = 5e-4 * optimizers.l2_norm(params)**2 # tf doesn't use sqrt
    # print('celoss', ce_loss)
    # print('l2_loss', l2_loss)
    return ce_loss

@jit
def l2_loss(params, batch):
    """
    The idxes of the batch indicate which nodes are used to compute the loss.
    """
    inputs, targets, adj, is_training, rng, idx = batch
    # preds = predict_fun(params, inputs, adj, is_training=is_training, rng=rng)
    # print(params)
    # ce_loss = -np.mean(np.sum(preds[np.array(idx)] * targets[np.array(idx)], axis=1))
    l2_loss = 5e-4 * optimizers.l2_norm(params)**2 # tf doesn't use sqrt
    # print('celoss', ce_loss)
    # print('l2_loss', l2_loss)
    return l2_loss   

@jit
def accuracy(params, batch):
    inputs, targets, adj, is_training, rng, idx = batch
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict_fun(params, inputs, adj, is_training=is_training, rng=rng), axis=1)
    return np.mean(predicted_class[np.array(idx)] == target_class[np.array(idx)])

@jit
def loss_accuracy(params, batch):
    inputs, targets, adj, is_training, rng, idx = batch
    preds = predict_fun(params, inputs, adj, is_training=is_training, rng=rng)
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(preds, axis=1)
    ce_loss = -np.mean(np.sum(preds[np.array(idx)] * targets[np.array(idx)], axis=1))
    acc = np.mean(predicted_class[np.array(idx)] == target_class[np.array(idx)])
    return ce_loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3409)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--pruning_method', type=str, default=None)
    parser.add_argument('--pruning_amount', type=float, default=50)
    parser.add_argument('--idx_noise_weight', type=float, default=0.0001)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sparse', dest='sparse', action='store_true')
    parser.add_argument('--no-sparse', dest='sparse', action='store_false')
    parser.add_argument('--eps', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.001)
    parser.set_defaults(sparse=True)
    args = parser.parse_args()

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, sparse = args.sparse)

    rng_key = random.PRNGKey(args.seed)
    dropout = args.dropout
    step_size = args.lr
    hidden = args.hidden
    num_epochs = args.epochs
    n_nodes = features.shape[0]
    n_feats = features.shape[1]
    early_stopping = args.early_stop

    clip_value = 1e-2
    pruning_method = args.pruning_method
    pruning_amount = args.pruning_amount
    index_noise_weigth = args.idx_noise_weight
    pruning_key = (pruning_method, index_noise_weigth, pruning_amount)
    epsilon = args.eps
    delta = args.delta

    init_fun, predict_fun = GCN(nhid=hidden, 
                                nclass=labels.shape[1],
                                dropout=dropout,
                                sparse=args.sparse)
    input_shape = (-1, n_nodes, n_feats)
    rng_key, init_key = random.split(rng_key)
    _, init_params = init_fun(init_key, input_shape)

    opt_init, opt_update, get_params = optimizers.adam(step_size)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        # print(grad(loss)(params, batch))
        # (grad, clip_value, pruning_method, epsilon, delta, train_epochs,)
        noisey_grad = perturb_grad(grad(loss)(params, batch),
                clip_value, pruning_key, epsilon, delta, num_epochs, i)
        summed_grad = jax.tree_util.tree_map(lambda x,y: x+y, noisey_grad, grad(l2_loss)(params, batch))
        return opt_update(i, summed_grad, opt_state)

    opt_state = opt_init(init_params)

    print("\nStarting training...")
    val_values = []
    for epoch in range(num_epochs):
        start_time = time.time()
        batch = (features, labels, adj, True, rng_key, idx_train)
        opt_state = update(epoch, opt_state, batch)
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        eval_batch = (features, labels, adj, False, rng_key, idx_val)
        train_batch = (features, labels, adj, False, rng_key, idx_train)
        train_loss, train_acc = loss_accuracy(params, train_batch)
        val_loss, val_acc = loss_accuracy(params, eval_batch)
        val_values.append(val_loss.item())
        print(f"Iter {epoch}/{num_epochs} ({epoch_time:.4f} s) train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # new random key at each iteration, othwerwise dropout uses always the same mask 
        rng_key, _ = random.split(rng_key)
        if epoch > early_stopping and val_values[-1] > numpy.mean(val_values[-(early_stopping+1):-1]):
            print("Early stopping...")
            break
    
    # now run on the test set
    test_batch = (features, labels, adj, False, rng_key, idx_test)
    test_acc = accuracy(params, test_batch)
    print(f'Test set acc: {test_acc}')