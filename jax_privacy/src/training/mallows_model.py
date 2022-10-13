from termios import N_SLIP
import chex 
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy
import time
import tensorflow as tf

class MallowsModel():
  def __init__(self, n, theta, pruning_amount, randomseed=None):
    if randomseed==None:
      numpy.random.seed()
      randomseed=numpy.random.randint(1,2147483647)
    self._random_key=jax.random.PRNGKey(randomseed)
    # self._random_key,_=jax.random.split(self._random_key)

    # === config that needs to be fixed before running ===

    # static params got from here
    self.pruning_amount=pruning_amount
    self.length=n//2
    self.pr_list=None
    self.last_pr_list=None
    self.theta=theta
    # self.change_num_list = jnp.load('/home/jungang/jax_privacy/jax_privacy/src/training/change_num_list.npy')
    # if self.change_num_list is None:
    #   self.init_pr_list(self.length, jnp.array(n*self.pruning_amount/100, int))
    self.change_num_list = None
    self.steps=0
    self.n = n
    
    '''
    # static params got from config file
    self.N=config.data.dataset.num
    self.pruning_amount=
    self.k=int(self.N*self.pruning_amount/100)
    '''

    # === ===
  # @partial(jit, static_argnums=1)
  def init_pr_list(self,n,k):
    k=jnp.min(jnp.array([k,n-k-1]))
    index=jnp.arange(self.length)
    ex=jnp.exp(-2*self.theta)
    multi=1/(index+1)
    multi=jnp.multiply(multi,multi)
    multi=jnp.multiply(multi,n-k-index)
    multi=jnp.multiply(multi,k-index)
    multi=multi*ex
    multi = jnp.append(jnp.array([1]), multi)
    # print(multi)
    pr = jnp.cumprod(multi)
    self.pr_list = pr/jnp.sum(pr)
    return self.pr_list

  def calculate_last_pr_list(self, n, k):
    k=jnp.min(jnp.array([k,n-k-1]))
    index=jnp.arange(self.length)
    ex=jnp.exp(-2*self.theta)
    multi=1/(index+1)
    multi=jnp.multiply(multi,multi)
    multi=jnp.multiply(multi,n-k-index)
    multi=jnp.multiply(multi,k-index)
    multi=multi*ex
    multi = jnp.append(jnp.array([1]), multi)
    # print(multi)
    pr = jnp.cumprod(multi)
    # print(pr)
    pr = pr / jnp.sum(pr)
    self.last_pr_list = pr
    return self.last_pr_list

  def sample_dist(self, n_list, n):
    self._random_key,_=jax.random.split(self._random_key)
    if self.change_num_list is not None and n % 256 == 0:
      self.steps = jnp.min(jnp.array([self.steps % 100000, (self.steps + n_list.size) % 100000]))
      num = jax.lax.dynamic_slice(self.change_num_list, (self.steps,), (n_list.size,))
      self.steps += n_list.size
    elif n % 256 == 0:
      num = jax.random.choice(self._random_key, self.length + 1, n_list.shape, p = self.pr_list)
    else:
      # print(random_num)
      if n_list.size ==1:
        # print(self.last_pr_list.shape)
        num = jax.random.choice(self._random_key, self.length + 1, p = self.last_pr_list)
      else:
        num = jax.random.choice(self._random_key, self.length + 1, n_list.shape, p = self.last_pr_list)
        # print(num)
    return num


  def perturb(self,mask,change_num):
    self._random_key,_=jax.random.split(self._random_key)
    random_mask = jax.random.uniform(key=self._random_key, shape=jnp.shape(mask))
    # print(random_mask.shape)
    # print(mask)
    mask_1 = random_mask * mask
    # print(mask_1.shape)
    # print('change_num', jnp.sum(change_num))
    if change_num.size < 2 and mask.shape[0] < 256:
      change_num = jnp.expand_dims(change_num,0)
    quantile_1 = jnp.percentile(jnp.abs(mask_1), 100-(change_num/((jnp.size(mask)/change_num.size))*100), axis=0)

    quantile_1 = jnp.expand_dims(jnp.diag(quantile_1), 0).repeat(mask.size // change_num.size, axis=0).reshape(mask.shape)

    mask_1 = jnp.where(jnp.abs(mask_1) >= quantile_1, 1, 0)
    
    mask_2 = random_mask * (jnp.ones_like(mask) - mask)
    quantile_2 = jnp.percentile(jnp.abs(mask_2), 100-(change_num/(jnp.size(mask)/change_num.size)*100), axis=0)
    quantile_2 = jnp.expand_dims(jnp.diag(quantile_2), 0).repeat(mask.size // change_num.size, axis=0).reshape(mask.shape)
    # print(quantile_1)
    mask_2 = jnp.where(jnp.abs(mask_2)> quantile_2, 1, 0)
    mask = mask - mask_1 + mask_2
    # print('mask_1',jnp.sum(mask_1))
    # print('change_num', change_num)
    # print(change_num-jnp.sum(mask_1, axis = 0))
    # print('mask_2',jnp.sum(mask_2))
    # print(jnp.sum(jnp.where(mask==1, 1,0)))
    # print(jnp.sum(jnp.abs(mask - mask_mod.reshape(mask.shape))))
    del mask_1, mask_2
    return mask

  def model(self, mask, pruning_amount):
    '''
    N=jnp.size(mask)
    list_len = int(N/256)+1
    '''
    n=self.n
    # start = time.time()
    if jnp.size(mask) % n == 0:
      n_list = jnp.ones(jnp.size(mask) // n) * n
      mask = mask.reshape(-1, mask.size // n)
      self.init_pr_list(n, jnp.array(n * pruning_amount/100, int))
      # print('mask_shape', mask.shape)
    else:
      if jnp.size(mask) < n:
        n_list = jnp.ones(1) * int(jnp.size(mask))
        self.calculate_last_pr_list(n_list, jnp.array(n_list* pruning_amount/100, int))
      elif jnp.size(mask) % 160 == 0:
        n = 160
        n_list = jnp.ones(jnp.size(mask) // n) * n
        mask = mask.reshape(-1, mask.size // n)
        self.init_pr_list(n, jnp.array(n * pruning_amount/100, int))
      elif jnp.size(mask) % 192 == 0:
        n = 192
        n_list = jnp.ones(jnp.size(mask) // n) * n
        mask = mask.reshape(-1, mask.size // n)
        self.init_pr_list(n, jnp.array(n * pruning_amount/100, int))
      else:
        mask = mask.reshape(-1, mask.size // n + 1)
        n_list = jnp.ones(mask.shape[1]) * int(mask.shape[0])
        self.calculate_last_pr_list(int(mask.shape[0]), jnp.array(mask.shape[0] * pruning_amount/100, int))
    
    percentile = jnp.expand_dims(jnp.percentile(jnp.abs(mask), 100 - pruning_amount, axis=0),0)
    percentile.repeat(mask.shape[0]).reshape(mask.shape)
    mask = jnp.where(jnp.abs(mask) > percentile, 1, 0)
    change_num = self.sample_dist(n_list, jnp.size(mask))
    mask=self.perturb(mask, change_num)
    return mask

  def random(self, mask, pruning_amount):
    n=self.n
    # start = time.time()
    if jnp.size(mask) % n == 0:
      n_list = jnp.ones(jnp.size(mask) // n) * n
      mask = mask.reshape(-1, mask.size // n)
      self.init_pr_list(n, jnp.array(n * pruning_amount/100, int))
      # print('mask_shape', mask.shape)
    else:
      if jnp.size(mask) < n:
        n_list = jnp.ones(1) * int(jnp.size(mask))
        self.calculate_last_pr_list(n_list, jnp.array(n_list* pruning_amount/100, int))
      elif jnp.size(mask) % 160 == 0:
        n = 160
        n_list = jnp.ones(jnp.size(mask) // n) * n
        mask = mask.reshape(-1, mask.size // n)
        self.init_pr_list(n, jnp.array(n * pruning_amount/100, int))
      elif jnp.size(mask) % 192 == 0:
        n = 192
        n_list = jnp.ones(jnp.size(mask) // n) * n
        mask = mask.reshape(-1, mask.size // n)
        self.init_pr_list(n, jnp.array(n * pruning_amount/100, int))
      else:
        mask = mask.reshape(-1, mask.size // n + 1)
        n_list = jnp.ones(mask.shape[1]) * int(mask.shape[0])
        self.calculate_last_pr_list(int(mask.shape[0]), jnp.array(mask.shape[0] * pruning_amount/100, int))
    
    percentile = jnp.expand_dims(jnp.percentile(jnp.abs(mask), 100 - pruning_amount, axis=0),0)
    percentile.repeat(mask.shape[0]).reshape(mask.shape)
    mask = jnp.where(jnp.abs(mask) > percentile, 1, 0)
    self._random_key,_=jax.random.split(self._random_key)
    random_mask = jax.random.uniform(key=self._random_key, shape=jnp.shape(mask))
    percentile = jnp.percentile(jnp.abs(random_mask), 100 - pruning_amount)
    random_mask = jnp.where(jnp.abs(random_mask) > percentile, 1, 0)
    return mask, random_mask
   

if __name__=='__main__':
  random_key_main=jax.random.PRNGKey(89)
  new_key,subkey=jax.random.split(random_key_main)
  random_key_main=new_key
  

  N=256*128
  pruning_amount=10
  k=int(N*pruning_amount/100)
  eps=1
  eps_mallows=0.01*eps
  per_example_pruning_amount=pruning_amount
  batch_size=4096
  data_num=50000
  total_step=875
  index_sen=N*per_example_pruning_amount
  q=batch_size/data_num
  eps_step=eps_mallows/total_step/q
  print(eps_step)
  theta=eps_step/index_sen
  begin_time = time.time()
  mallows_model=MallowsModel(256, theta, pruning_amount)
  grad=jax.random.uniform(random_key_main,(int(N),))
  t=jnp.percentile(grad,100-pruning_amount)
  mask_0=grad
  end_time = time.time()
  # print('mask0',jnp.sum(mask_0))
  
  # for j in range(2):
  dist_1 = []
  dist_2 = []
  for i in range(300):
    mask_mod=mallows_model.model(mask_0, pruning_amount)
    mask, random_mask = mallows_model.random(mask_0, pruning_amount)
    print(jnp.sum(random_mask))
    # n_list = jnp.full(100000, 256)
    # n_list = jnp.array_split(n_list, 10000)
    # print(n_list.shape)
    # change_num_list = jax.tree_util.tree_map(lambda x: mallows_model.sample_dist(x), n_list)
    # print(change_num_list.shape)
    # print(change_num_list[10000:10050])
    # jnp.save('change_num_list__', change_num_list)
    
    print(jnp.sum(mask_mod))
    print('dist_1',numpy.array(jnp.sum(jnp.abs(mask - mask_mod))))
    print('dist_2', numpy.array(jnp.sum(jnp.abs(mask - random_mask))))
    dist_1.append(numpy.array(jnp.sum(jnp.abs(mask - mask_mod))).tolist())
    dist_2.append(numpy.array(jnp.sum(jnp.abs(mask - random_mask))).tolist())
    # tf.summary.histogram('Mallows dist', numpy.array(dist_1), j)
    # tf.summary.histogram('Random dist', numpy.array(dist_2), j)

  # print(jnp.sum(jnp.abs(mask_0 - mask_mod.reshape(mask_0.shape))) )
  print('total time ', end_time - begin_time)

  from matplotlib import pyplot as plt
  import matplotlib.patches as patches
  import seaborn as sn
  import json
  import pandas as pd

  # with open('./data1.json', 'r') as f:
  #     index = json.load(f)

  # with open('./data2.json', 'r') as f:
  #     index_b = json.load(f)
  index = dist_1
  index_b = dist_2

  d = {'': ['Distance between Mallows Model Perturbed Index and I0' for i in index]+['Distance between Random Index and I0' for i in index_b], 'Distance': index+index_b}
  df = pd.DataFrame(data=d)

  sn.set_style('ticks')
  sn.set_context("paper", font_scale=2.2, rc={'line.linewidth':3})
  sn.color_palette("husl")
  fig=plt.figure(figsize=(10,8))
  plt.xlabel('Distance', fontsize=30)
  plt.ylabel('Density Distribution', fontsize=30)

  plt.xticks(fontsize=10)
  plt.yticks(fontsize=20)
  sn.histplot(df, x='Distance', bins=25, kde=True, stat="density", hue='', common_norm=False)

  ax=plt.gca()
  rect=patches.Rectangle((2150, 0),360,0.0032,linewidth=2, edgecolor='r',facecolor='none', linestyle='--')
  ax.add_patch(rect)
  plt.show()
  plt.savefig('./distribution_256.jpg', bbox_inches='tight')