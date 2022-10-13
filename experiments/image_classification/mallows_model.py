import chex 
import jax
import jax.numpy as jnp

class MallowsModel():
  def __init__(self):
    self._random_key=jax.random.PRNGKey(144)

  def sample_dist(self,theta,n,k):
    self._random_key,_=jax.random.split(self._random_key)
    random_num=jax.random.uniform(self._random_key)
    k=jnp.where(k > n-k, k, n-k)[0]
    pr=1.
    pr_sum=0.
    pr_list=jnp.empty(k)
    index=jnp.arange(k)
    ex=jnp.exp(-2*theta)
    multi=1/(index+1)
    multi=jnp.multiply(multi,multi)
    multi=jnp.multiply(multi,n-k-index)
    multi=jnp.multiply(multi,k-index)
    for i in range(k):
      pr_sum+=pr
      pr*=ex*multi[i]
      pr_list=pr_list.at[i].set(pr_sum)
    pr_list=pr_list/pr_sum
    # print('prlist',pr_list)
    for i in range(k):
      if random_num<=pr_list[i]:
        if i == 0:
          return 0
        else:
          return i-1

  def perturb(self,mask,change_num):
    # bug here
    index_1=jnp.where(mask==1)[0]
    self._random_key,_=jax.random.split(self._random_key)
    set_0=jax.random.choice(self._random_key,index_1,(change_num,))
    index_0=jnp.where(mask==0)[0]
    self._random_key,_=jax.random.split(self._random_key)
    set_1=jax.random.choice(self._random_key,index_0,(change_num,))
    for x in set_0:
      mask=mask.at[x].set(0)
    for x in set_1:
      mask=mask.at[x].set(1)
    # bug above
    return mask

  def model(self,mask,theta,pruning_amount, N):
    # N=jnp.array(N, int)
    list_len = jnp.array((N/256+1), int)
    n_list = jnp.ones(100, dtype=int) * 256
    n_list = n_list.at[list_len - 1].set(N - (list_len - 1) * 256)
    n_list = list(n_list)
    # print(n_list)
    change_num_list = jax.tree_util.tree_map(lambda x: self.sample_dist(theta, jnp.array(x, int), jnp.array(x*pruning_amount/100, int)), n_list)
    # print('change_list', change_num_list)
    change_num=jnp.sum(jnp.array(change_num_list))
    # print(change_num)
    mask_1=self.perturb(mask,change_num)
    return mask_1
   

if __name__=='__main__':
  random_key_main=jax.random.PRNGKey(42)
  new_key,subkey=jax.random.split(random_key_main)
  random_key_main=new_key
  
  mallows_model=MallowsModel()
  N=1000
  pruning_amount=90
  k=int(N*pruning_amount/100)
  eps=1
  eps_mallows=0.1*eps
  per_example_pruning_amount=pruning_amount*0.1
  batch_size=4096
  data_num=50000
  total_step=875
  index_sen=N*per_example_pruning_amount
  q=batch_size/data_num
  eps_step=eps_mallows/total_step/q
  print(eps_step)
  theta=eps_step/index_sen/int(N/256)
  grad=jax.random.uniform(random_key_main,(int(N),))
  t=jnp.percentile(grad,100-pruning_amount)
  mask_0=jnp.where(grad>t,1,0)
  print(jnp.sum(mask_0))
  mask_mod=mallows_model.model(mask_0,theta,pruning_amount,N)
  print(jnp.sum(mask_mod))

