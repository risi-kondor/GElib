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

*******
SO3part
*******

An ``SO3part`` objects stores `m` different vectors transforming accoring to the same 
irreducible representation :math:`D^{(\ell)}`. 
Since :math:`D^{(\ell)}` is a :math:`2\ell\!+\!1` dimensional representation, 
the vectors can be jointly stored in a matrix :math:`\mathbb{C}^{(2\ell+1)\times n}`.   

The following constructs an ``SO3part`` object holding ``n=3`` vectors corresponding to the ``l=2`` 
irrep. 

.. code-block:: python

 >>> A=SO3part.gaussian(2,3)
 >>> A
 GElib::SO3part(l=2,n=3)
 >>> print(A)
 [ (-0.302782,0.137796) (-0.999364,0.674001) (-1.62807,-0.281158) ]
 [ (-0.250657,-1.1945) (0.658802,1.06918) (-0.973458,-1.2115) ]
 [ (-1.20183,-2.1947) (-0.399454,0.680457) (-0.727057,1.40633) ]
 [ (0.43853,-0.31774) (-0.42954,0.123568) (-2.20967,-0.37048) ]
 [ (-1.22569,0.0266631) (0.73464,0.592888) (0.630166,-0.165017) ]

=====================
Access and arithmetic
=====================

 
``SO3part`` objects support the usual arithmetic operations of addition, subtraction, multiplication 
by scalars, and so on. 

.. code-block:: python

 >>> A=SO3part.gaussian(2,3)
 >>> B=SO3part.gaussian(2,3)
 >>> print(A+B)
 [ (1.47879,-0.459842) (0.490458,0.545539) (1.19309,-0.792573) ]
 [ (2.22926,-3.41237) (-2.06763,1.01224) (-2.15526,-0.777298) ]
 [ (-0.653524,-0.00262699) (-0.422077,0.3195) (-1.5863,1.06357) ]
 [ (-1.99061,-2.98562) (1.96568,0.418992) (3.30987,0.511894) ]
 [ (-0.00645578,-1.14759) (1.30118,-0.668608) (-2.32167,-1.85946) ]

 >>> print(A-B)
 [ (1.99449,1.53405) (1.27393,0.844093) (-4.19867,1.87504) ]
 [ (-1.08774,-0.525339) (0.207749,-0.303882) (0.285285,-0.133225) ]
 [ (-0.875828,0.758615) (0.923785,-0.867636) (1.20997,-1.05234) ]
 [ (-1.03569,-0.560109) (0.679434,0.620389) (0.55949,-0.425507) ]
 [ (2.51134,-2.78578) (0.782226,-0.292865) (0.927737,-1.81336) ]

 >>> print(3.0*A)
 [ (5.20991,1.61131) (2.64658,2.08445) (-4.50837,1.62369) ]
 [ (1.71228,-5.90657) (-2.78982,1.06253) (-2.80496,-1.36578) ]
 [ (-2.29403,1.13398) (0.752563,-0.822204) (-0.564493,0.016848) ]
 [ (-4.53945,-5.31859) (3.96768,1.55907) (5.80405,0.12958) ]
 [ (3.75732,-5.90005) (3.12511,-1.44221) (-2.09089,-5.50922) ]


Individual entries in an ``SO3part`` can be accessed with the same syntax as how matrix elements are accessed. 
Note however the that indexing convention is that ``n`` index comes first, and the ``m`` index corresponding 
to indexing the components within a given irreducible vector is second. 

It is also important to note that in general manually setting entries in ``SO3part`` objects individually 
breaks equivariance. 

.. code-block:: python

 >>> A=SO3part.gaussian(2,3)
 >>> A(2,3)
 (1.4733880758285522+0.1949467808008194j)
 >>> A[2,3]
 (1.4733880758285522+0.1949467808008194j)
 >>> A[2,3]=1.0
 >>> print(A)
 [ (-1.23974,1.16493) (-0.407472,0.584898) (1.61201,-0.660558) ]
 [ (0.399771,0.534755) (1.3828,-0.607787) (0.0523187,0.74589) ]
 [ (-0.904146,-1.75177) (1.87065,-0.965146) (-1.66043,-0.474282) ]
 [ (-0.688081,-0.546571) (0.0757219,-0.0384917) (1,0) ]
 [ (0.097221,-0.485144) (-0.89237,-0.370271) (-0.228782,-1.12408) ]

=================
Matrix operations
=================


``SO3part`` objects can also be multiplied by ``cnine::ctensor`` matrices from the `right`. 
Multiplication by matrices from the left is not allowed because it would break equivariance. 
Note that to construct ``cnine`` tensors, first the ``cnine`` module must be loaded.

.. code-block:: python

 >>> from GElib import *
 >>> A=SO3part.gaussian(2,3)
 >>> from cnine import *
 >>> M=ctensor.gaussian([3,3])
 >>> B=A*M
 >>> print(B)
 [ (6.03209,0.449935) (-0.717159,1.11423) (-4.45347,-1.60968) ]
 [ (-1.65973,0.693803) (1.65098,-1.79472) (2.36366,-1.61901) ]
 [ (-2.75144,0.117771) (1.33895,-3.43707) (9.36576,1.70359) ]
 [ (2.5369,3.65761) (-0.0298907,-0.733082) (-0.894675,-1.63206) ]
 [ (1.22471,-1.53125) (-1.74749,0.0933496) (-1.33813,1.35488) ]

===================
Spherical harmonics
===================


``SO3part`` objects can be initialized as spherical harmonic coefficients of a three dimensional vector 
correspnding to a given :math:`\ell`.  

.. code-block:: python

 >>> A=SO3part.spharm(2,[0.12,0.31,-0.55])
 >>> print(A)
 [ (-0.0764131,-0.0695855) ]
 [ (-0.123458,0.318933) ]
 [ (0.37763,0) ]
 [ (0.123458,0.318933) ]
 [ (-0.0764131,0.0695855) ]

==============
GPU operations
==============


Similarly to ``cnine`` tensors, ``SO3part`` objects can moved back and forth between the host (CPU) 
and the GPU with the ``to`` method. 

.. code-block:: python

  >>> A=SO3part.gaussian(4,4)
  >>> B=A.to(1) # Create a copy of A on the first GPU (GPU0)
  >>> C=B.to(0) # Move B back to the host 


