import chex
import jax
import jax.numpy as jnp
import optax


def cos_sim(
    ori_grad: chex.ArrayTree,
    noise_grad: chex.ArrayTree,
) -> chex.Numeric:
  # print(ori_grad)
  tree_value,_ =jax.tree_util.tree_flatten(ori_grad)
  noise_tree_value,_ =jax.tree_util.tree_flatten(noise_grad)
  # print('noise',tree_value.reshape(-1))
  # print('grad',noise_tree_value)
  product = jnp.array(list(map(lambda x, y: jnp.dot(x.reshape(-1), y.reshape(-1)), tree_value, noise_tree_value)))
  # print('product',product)
  cos = product.sum() / optax.global_norm(ori_grad) / optax.global_norm(noise_grad)
  
  return cos

if __name__=='__main__':
  a = jnp.array([5, 7, 9], dtype=float)
  b = jnp.array([6, 1, 2], dtype=float)
  c = jnp.array([6, 1, 1], dtype=float)
  tree = [a, b]
  tree_2 = [a, c]
  scalar = dict(cos=a,b=b)
  print(scalar['b'], scalar['cos'])
  # cos = cos_sim(a, b)
  cos = cos_sim(tree, tree_2)
  print(cos)
