import numpy as np
from scipy.special import comb
import time

def sample_dist(theta, n, k):
  t = np.random.rand()
  # print(t)
  pr_list = []
  pr_sum = 0.
  k = np.min([k, n-k])
  pr = 1.
  for i in range(k):
    pr_sum += pr
    pr *= np.exp(-2*theta)/(i+1) /(i+1) * (n- k- i) * (k- i)
    pr_list.append(pr_sum)
  # print(pr_sum)
  pr_list = pr_list / pr_sum
  # print(pr_list)
  for i in range(k):
    if t <= pr_list[i]:
      return i-1


def perturb(mask, change_num):
  index_1 =np.where(mask == 1)
  # print(index_1[0])
  set_0 = np.random.choice(index_1[0], change_num)
  # print(set_0)
  index_0 = np.where(mask == 0)
  set_1 = np.random.choice(index_0[0], change_num)
  mask[set_0] = 0
  # print(mask_0.sum())
  mask[set_1] = 1
  # print(mask_0.sum())
  return mask

def Mallows_model(mask, theta, pruning_amount):
  N = np.size(mask)
  change_num = 0
  while N >= 1:
    if N <=256:
      n = N
    else:
      n = 256
    k = int(n * pruning_amount / 100)
    change_num += sample_dist(theta, n, k)
    N -= 256
  print(change_num)
  mask_1 = perturb(mask, change_num)
  return mask_1


N = 1e7
pruning_amount = 50
k = int(N * pruning_amount / 100)
eps = 1
eps_mallows = 0.1*eps
per_example_pruning_amount = pruning_amount*0.1
batch_size = 4096
data_num = 50000
total_steps = 875
index_sen = N * per_example_pruning_amount
q = batch_size / data_num
eps_step = eps_mallows/total_steps/q
print(eps_step)

theta = eps_step / index_sen / (int(N/256)+1)

grad = np.random.randn(int(N))
t = np.percentile(np.abs(grad), 100-pruning_amount)
mask_0 = np.where(np.abs(grad)>t, np.ones_like(grad), np.zeros_like(grad))
print(np.sum(mask_0))
start = time.time()
mask = Mallows_model(mask_0, theta,  pruning_amount)
end = time.time()
print(np.sum(mask))
print(end - start)
print((end - start)*875)


