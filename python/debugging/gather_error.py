import torch.nn as nn
import cnine
from gelib import *
import numpy as np

# define our array of SO3 vectors
L = 2 # max l
n_ch = 2 # channels
n_a = 2 # atoms
tau = n_ch*np.ones((L+1)).astype(int)
random_SO3partArr = SO3partArr.randn(1,[n_a],1,2)
random_SO3vecArr = SO3vecArr.randn(1,[n_a],tau)

# define our random connectivity matrix
random_connectivity = torch.randn(n_a,n_a)

# do gather
mask = cnine.Rmask1(random_connectivity)
print(mask)

# gathered_SO3partArr = random_SO3partArr.gather(mask)
gathered_SO3vecArr = random_SO3vecArr.gather(mask)

#print("\n \033[1m A random part was defined as \n \033[0m", random_SO3partArr)
#print("\n \033[1m The gather operation was performed with connectivity \n \033[0m", mask)
#print("\n \033[1m The output is \n \033[0m", gathered_SO3partArr)

print("\n \033[1m A random vector was defined as \n \033[0m", random_SO3vecArr)
print("\n \033[1m The gather operation was performed with connectivity \n \033[0m", mask)
print("\n \033[1m The output is \n \033[0m", gathered_SO3vecArr)
