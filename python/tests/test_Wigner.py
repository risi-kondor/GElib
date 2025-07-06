import numpy as np
import gelib as g
import torch
from lie_learn.representations.SO3.wigner_d import wigner_D_matrix


l=1

for it in range(0,5):
    
    alpha=np.random.uniform(0,2*np.pi)
    beta=np.random.uniform(0,np.pi)
    gamma=np.random.uniform(0,2*np.pi)

    D=wigner_D_matrix(l, alpha, beta, gamma, field='complex')
    print(D)

    rho=g.SO3irrep(l)
    Dd=rho.matrix([alpha,beta,gamma])
    print(Dd)

    print(torch.allclose(torch.from_numpy(D).cfloat(),Dd,rtol=1e-6))
    print("\n")

    
