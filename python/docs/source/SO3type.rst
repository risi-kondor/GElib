*******
SO3type
*******

An ``SO3type`` object specifies the multiplicites :math:`\tau_0,\tau_1,\ldots,\tau_L` of each irreducible 
representation :math:`D^{(0)},D^{(1)},\ldots,D^{(L)}` in an SO3-vector. 

.. code-block:: python

 >>> tau=SO3type([3,4,1]) # Define tau=(3,4,1)
 >>> tau
 <GElib::SO3type(3,4,1)>
 >>> print(tau)
 (3,4,1)
 >>> tau[2]               # Get tau_2
 1
 >>> tau[2]=3             # Set tau_2=3
 >>> tau
 <GElib::SO3type(3,4,3)>
 >>> tau.maxl()           # Print out the index of the largest irrep 
 2

The function ``CGproduct`` computes the type of the Clebsch--Gordan product of two SO-vectors 
of given types.

.. code-block:: python

 >>> tau=SO3type([2,2])
 >>> mu=CGproduct(tau,tau)
 >>> print(mu)
 (8,12,4)

The optional argument `maxl` limits the product to a given maximal :math:`\ell`. 

.. code-block:: python

 >>> tau=SO3type([1,1,1])
 >>> print(CGproduct(tau,tau))
 (3,6,6,3,1)
 >>> print(CGproduct(tau,tau,2))
 (3,6,6)

