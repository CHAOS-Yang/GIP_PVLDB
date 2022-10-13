import chex 
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy
import time
import os

class MallowsModel():
  def __init__(self, theta, randomseed=None):
    if randomseed==None:
      numpy.random.seed()
      randomseed=numpy.random.randint(1,2147483647)
    self._random_key=jax.random.PRNGKey(randomseed)

    # === config that needs to be fixed before running ===

    # static params got from here
    self.theta = theta
    self.length=128
    self.pr_list=None
    self.init_prop(256)
    '''
    # static params got from config file
    self.N=config.data.dataset.num
    self.pruning_amount=
    self.k=int(self.N*self.pruning_amount/100)
    '''

  def merge(left, right):
    """
    This function uses Merge sort algorithm to count the number of
    inversions in a permutation of two parts (left, right).
    Parameters
    ----------
    left: ndarray
        The first part of the permutation
    right: ndarray
        The second part of the permutation
    Returns
    -------
    result: ndarray
        The sorted permutation of the two parts
    count: int
        The number of inversions in these two parts.
    """
    result = []
    count = 0
    i, j = 0, 0
    left_len = len(left)
    while i < left_len and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            count += left_len - i
            j += 1
    result += left[i:]
    result += right[j:]

    return result, count

  def mergeSort_rec(self, lst):
      """
      This function splits recursively lst into sublists until sublist size is 1. Then, it calls the function merge()
      to merge all those sublists to a sorted list and compute the number of inversions used to get that sorted list.
      Finally, it returns the number of inversions in lst.
      Parameters
      ----------
      lst: ndarray
          The permutation
      Returns
      -------
      result: ndarray
          The sorted permutation
      d: int
          The number of inversions.
      """
      lst = list(lst)
      if len(lst) <= 1:
          return lst, 0
      middle = int( len(lst) / 2 )
      left, a   = self.mergeSort_rec(lst[:middle])
      right, b  = self.mergeSort_rec(lst[middle:])
      sorted_, c = self.merge(left, right)
      d = (a + b + c)
      return sorted_, d
      
  def distance(self, A, B=None):
      """
      This function computes Kendall's-tau distance between two permutations
      using Merge sort algorithm.
      If only one permutation is given, the distance will be computed with the
      identity permutation as the second permutation
    Parameters
    ----------
    A: ndarray
          The first permutation
    B: ndarray, optional
          The second permutation (default is None)
    Returns
    -------
    int
          Kendall's-tau distance between both permutations (equal to the number of inversions in their composition).
      """
      if B is None : B = list(range(len(A)))

      A = jnp.asarray(A).copy()
      B = jnp.asarray(B).copy()
      n = len(A)

      # check if A contains NaNs
      msk = jnp.isnan(A)
      indexes = jnp.array(range(n))[msk]

      if indexes.size:
          A = A.at[indexes].set(n)#jnp.nanmax(A)+1

      # check if B contains NaNs
      msk = jnp.isnan(B)
      indexes = jnp.array(range(n))[msk]

      if indexes.size:
          B = B.at[indexes].set(n)#jnp.nanmax(B)+1

      # print(A,B,n)
      inverse = jnp.argsort(B)
      compose = A[inverse]
      _, distance = self.mergeSort_rec(compose)
      return distance


  def max_dist(n):
      """ This function computes the maximum distance between two permutations of n length.
          Parameters
          ----------
          n: int
              Length of permutations
          Returns
          -------
          int
              Maximum distance between permutations of given n length.
      """
      return int(n*(n-1)/2)


  #************ Vector/Rankings **************#

  def v_to_ranking(self, v, n):
      """This function computes the corresponding permutation given a decomposition vector.
      The O(n log n) version in 10.1.1 of
      Arndt, J. (2010). Matters Computational: ideas, algorithms, source code.
      Springer Science & Business Media.
          Parameters
          ----------
          v: ndarray
              Decomposition vector, same length as the permutation, last item must be 0
          n: int
              Length of the permutation
          Returns
          -------
          ndarray
              The permutation corresponding to the decomposition vectors.
      """
      print('v', v)
      print('n', n)
      rem = list(range(n))
      rank = jnp.full(n, jnp.nan)
      for i in range(len(v)):
          rank = rank.at[i].set(rem[v[i]])
          rem.pop(v[i])
      return rank.astype(int)

  def ranking_to_v(self, sigma, k=None):
      """This function computes the corresponding decomposition vector given a permutation
      The O(n log n) version in 10.1.1 of
      Arndt, J. (2010). Matters Computational: ideas, algorithms, source code.
      Springer Science & Business Media.
          Parameters
          ----------
          sigma: ndarray
              A permutation
          k: int, optional
              The index to perform the conversion for a partial
              top-k list
          Returns
          -------
          ndarray
              The decomposition vector corresponding to the permutation. Will be
              of length n and finish with 0.
      """
      n = len(sigma)
      if k is not None:
          sigma = sigma[:k]
          sigma = jnp.concatenate((sigma, jnp.array([jnp.float(i) for i in range(n) if i not in sigma])))
      V = []
      for j, sigma_j in enumerate(sigma):
          V_j = 0
          for i in range(j+1, n):
              if sigma_j > sigma[i]:
                  V_j += 1
          V.append(V_j)
      return jnp.array(V)


  #************ Sampling ************#
  def init_prop(self, n):
      theta = jnp.full(n-1, self.theta) 
      rnge = jnp.array(range(n-1))
      phi = jnp.exp(-self.theta)
      print(rnge)
      psi_inv = (1 - jnp.exp( -theta[rnge]))/(1 - jnp.exp(( - n + rnge )*(theta[ rnge ])))
      psi_inv = jnp.append(psi_inv, jnp.array([1]))
      print(psi_inv.shape)
      self.vprobs = jnp.zeros((n, n))
      self.vprobs = self.vprobs.at[rnge,0].set(1.0*psi_inv[rnge])
      self.vprobs = self.vprobs.at[n-1,0].set(1.0)
      print(self.vprobs)
      for j in range(n-1):
          self.vprobs = self.vprobs.at[j,:n-j-1].set(jnp.exp( -theta[:n-j-1]))
      self.vprobs = self.vprobs * psi_inv
      print(self.vprobs)

  def sample(self, m, n, *, k=None, theta=None, phi=None, s0=None):
      """This function generates m (rankings) according to Mallows Models (if the given parameters
      are m, n, k/None, theta/phi: float, s0/None) or Generalized Mallows Models (if the given
      parameters are m, n, theta/phi: ndarray, s0/None). Moreover, the parameter k allows the
      function to generate top-k rankings only.
          Parameters
          ----------
          m: int
              Number of rankings to generate
          n: int
              Length of rankings
          theta: float or ndarray, optional (if phi given)
              The dispersion parameter theta
          phi: float or ndarray, optional (if theta given)
              Dispersion parameter phi
          k: int
              Length of partial permutations (only top items)
          s0: ndarray
              Consensus ranking
          Returns
          -------
          ndarray
              The rankings generated
      """

      # theta, phi = mm.check_theta_phi(theta, phi)

      theta = jnp.full(n-1, self.theta)

      if s0 is None:
          s0 = jnp.array(range(n))

      
      sample = []
      vs = []
      for samp in range(m):
          v = [jax.random.choice(self._random_key, n, p=self.vprobs[i, :]) for i in range(n-1)]
          v += [0]
          ranking = self.v_to_ranking(v,n)
          sample.append(ranking)

      sample = jnp.array([s[s0] for s in sample])

      if k is not None:
          sample_rankings = jnp.array([jnp.argsort(ordering) for ordering in sample])
          sample_rankings = jnp.array([ran[s0] for ran in sample_rankings])
          sample = jnp.array([[i if i in range(k) else jnp.nan for i in ranking] for
                          ranking in sample_rankings])
      return sample

  # def perturb(self,mask,change_num):
  #   self._random_key,_=jax.random.split(self._random_key)
  #   random_mask = jax.random.uniform(key=self._random_key, shape=jnp.shape(mask))
  #   mask_1 = random_mask * mask
  #   quantile_1 = jnp.percentile(jnp.abs(mask_1), 100-(change_num/jnp.size(mask)*100))
  #   mask_1 = jnp.where(jnp.abs(mask_1)> quantile_1, jnp.ones_like(mask_1), jnp.zeros_like(mask_1))
  #   mask_2 = random_mask * (jnp.ones_like(mask) - mask)
  #   quantile_2 = jnp.percentile(jnp.abs(mask_2), 100-(change_num/jnp.size(mask)*100))
  #   # print(quantile_2)
  #   mask_2 = jnp.where(jnp.abs(mask_2)> quantile_2, jnp.ones_like(mask_2), jnp.zeros_like(mask_2))
  #   # print(jnp.sum(mask_1))
  #   # print(jnp.sum(mask_1+mask_2))
  #   mask = mask - mask_1 + mask_2
  #   return mask

  # def model(self,mask,theta,pruning_amount):
  #   '''
  #   N=jnp.size(mask)
  #   list_len = int(N/256)+1
  #   '''
  #   n=256
  #   start = time.time()
  #   n_list = jnp.ones(int(jnp.size(mask)/n)+1) * n
  #   n_list = n_list.at[int(jnp.size(mask)/n)].set(jnp.size(mask) - (int(jnp.size(mask)/n)) * n)
  #   print(n_list)
  #   n_last = jnp.size(mask) - (int(jnp.size(mask)/n)) * n
  #   stop_1 = time.time()
  #   self.init_pr_list(theta, n, jnp.array(n*pruning_amount/100, int))
  #   stop_2 = time.time()
  #   self.calculate_last_pr_list(theta, n_last, jnp.array(n_last*pruning_amount/100, int))
  #   stop_3 = time.time()
  #   change_num_list = self.sample_dist(n_list)
  #   change_num=jnp.sum(jnp.array(change_num_list))
  #   print('change_num',change_num)
  #   stop_4 = time.time()
  #   mask_1=self.perturb(mask,change_num)
  #   stop_5 = time.time()
  #   print('time list')
  #   print(stop_1 - start)
  #   print(stop_2 - stop_1)
  #   print(stop_3 - stop_2)
  #   print(stop_4 - stop_3)
  #   print(stop_5 - stop_4)
  #   # print(mask_1 - mask)
  #   return mask_1
   

if __name__=='__main__':
  random_key_main=jax.random.PRNGKey(89)
  new_key,subkey=jax.random.split(random_key_main)
  random_key_main=new_key
  
  
  N=256
  pruning_amount=10
  k=int(N*pruning_amount/100)
  eps=1
  eps_mallows=0.1*eps
  per_example_pruning_amount=pruning_amount
  batch_size=4096
  data_num=50000
  total_step=875
  index_sen=N*per_example_pruning_amount
  q=batch_size/data_num
  eps_step=eps_mallows/total_step/q
  print(eps_step)
  theta=eps_step/index_sen
  mallows_model=MallowsModel(theta)
  grad=jax.random.uniform(random_key_main,(int(N),))
  t=jnp.percentile(grad,100-pruning_amount)
  mask_0=jnp.where(grad>t,1,0)
  print('mask0',jnp.sum(mask_0))
  sigma = mallows_model.sample(1,N)
  mask_mod = mask_0.at[sigma].set(mask_0)
  print(jnp.sum(jnp.abs(mask_mod-mask_0))/2/jnp.sum(mask_0))

