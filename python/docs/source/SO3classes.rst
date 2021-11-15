###########
SO3 classes
###########


The fundamental classes in GElib for storing and manipulating SO(3)-equivariant vectors are ``SO3vec``, 
``SO3part`` and ``SO3type``. 

``SO3vec`` stores a vector that the group SO(3) acts on. 
Any such vector can be decomposed into a sequence of smaller vectors that :math:`\mathrm{SO}(3)` 
acts on in an irreducible way, i.e., vectors that transform as 
:math:`\mathbf{v}^{(\ell)}_i\mapsto D^{(l)}(R)\,\mathbf{v}^{(\ell)}_i` for one of the irreducible 
representations :math:`D^{(l)}(R)` of :math:`\mathrm{SO}(3)`. 

``SO3part`` collects those components of an SO3-vector that transform according to the *same* irreducible 
representation. 
Finally, ``SO3type`` is a vector of integers that specifies how many irreducible components corresponding to 
:math:`D^{(0)},D^{(1)},\ldots` are featured in a given SO3-vector. 

*******
SO3type
*******

An ``SO3type`` object specifies the multiplicites :math:`\tau_0,\tau_1,\ldots,\tau_L` of each irreducible 
representation :math:`D^{(0)},D^{(1)},\ldots,D^{(L)}`. 

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

