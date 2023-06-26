*******
SO3part
*******

An ``SO3part`` objects stores `n` different vectors transforming accoring to the same 
irreducible representation :math:`D^{(\ell)}`. 
Since :math:`D^{(\ell)}` is a :math:`2\ell\!+\!1` dimensional representation, 
the vectors are jointly stored in a matrix :math:`\mathbb{C}^{(2\ell+1)\times n}`.   

The following constructs an ``SO3part`` object holding ``n=3`` gaussian random vectors corresponding to the ``l=2`` 
irrep. 

.. code-block:: python

 >>> A=gelib.SO3part.gaussian(2,3)
 >>> A
 GElib::SO3part(l=2,n=3)
 >>> print(A)
 [ (-0.302782,0.137796) (-0.999364,0.674001) (-1.62807,-0.281158) ]
 [ (-0.250657,-1.1945) (0.658802,1.06918) (-0.973458,-1.2115) ]
 [ (-1.20183,-2.1947) (-0.399454,0.680457) (-0.727057,1.40633) ]
 [ (0.43853,-0.31774) (-0.42954,0.123568) (-2.20967,-0.37048) ]
 [ (-1.22569,0.0266631) (0.73464,0.592888) (0.630166,-0.165017) ]

Similarly to real and complex valued tensors in `cnine`, ``SO3part`` objects can also be created with the 
`zero`, `ones` and `raw` constructors. The optional ``device`` argument specifies with the SO3part 
is created on the CPU or the GPU.

The order ``l`` and multiplicity ``n`` of an ``SO3part`` is accessed as follows.

.. code-block:: python

 >>> A=gelib.SO3part.gaussian(2,3)
 >>> A.getl()
 2
 >>> A.getn()
 3
 

=====================
Access and arithmetic
=====================

 
``SO3part`` objects support the usual arithmetic operations of addition, subtraction, multiplication 
by scalars, and so on. 

.. code-block:: python

 >>> A=gelib.SO3part.gaussian(2,3)
 >>> B=gelib.SO3part.gaussian(2,3)
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


Individual entries in an ``SO3part`` can be accessed with the same syntax as how matrix elements are accessed 
in ``cnine``. 
Note however the that indexing convention is that ``n`` index comes first, and the ``m`` index 
:math:`0\leq m< 2\ell+1` corresponding to the index within a given irreducible vector is second.  
It is important to note that manually setting entries in ``SO3part`` objects individually in general 
breaks equivariance. 

.. code-block:: python

 >>> A=gelib.SO3part.gaussian(2,3)
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

============
Group action
============

SO(3) acts on an ``SO3part`` :math:`A` of index :math:`\ell` by :math:`A\mapsto D^{(l)}(r) A` 
for any :math:`r\in\mathrm{SO}(3)`. This action is implemented via the ``apply`` method.  

.. code-block:: python

  >>> A=gelib.SO3part.gaussian(2,2)
  [ (-1.23974,0.0757219) (-0.407472,1.47339) ]
  [ (1.61201,0.097221) (0.399771,-0.89237) ]
  [ (1.3828,-0.228782) (0.0523187,1.16493) ]
  [ (-0.904146,0.584898) (1.87065,-0.660558) ]
  [ (-1.66043,0.534755) (-0.688081,-0.607787) ]
  >>>
  >>> r=gelib.SO3element.uniform()
  >>> Ar=A.apply(r)
  >>> print(Ar)
  [ (-1.11709,0.147657) (-1.11769,1.24745) ]
  [ (-1.60924,0.32233) (1.23518,1.00509) ]
  [ (0.618006,-0.476802) (-0.583183,-0.775842) ]
  [ (2.06776,-0.517233) (0.637578,0.790121) ]
  [ (-1.09639,0.00411239) (0.878735,-1.1598) ]


======================
Functions of SO3-parts
======================

Similarly to tensors, it is possible to take the inner products and norms of ``SO3part`` objects. 

.. code-block:: python

 >>> A=gelib.SO3part.gaussian(2,3)
 >>> B=gelib.SO3part.gaussian(2,3)
 >>> gelib.inp(A,B)
 (1.5953152179718018+5.115486145019531j)
 >>> gelib.norm2(A)
 7.411661148071289


=================
Matrix operations
=================


``SO3part`` objects can be multiplied by ``cnine::ctensor`` matrices from the `right`. 
Multiplication by matrices from the left is not allowed because it would break equivariance. 
Note that to construct ``cnine`` tensors, first the ``cnine`` module must be loaded.

.. code-block:: python

 >>> from cnine import *
 >>> A=gelib.SO3part.gaussian(2,3)
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

 >>> A=gelib.SO3part.spharm(2,[0.12,0.31,-0.55])
 >>> print(A)
 [ (-0.0764131,-0.0695855) ]
 [ (-0.123458,0.318933) ]
 [ (0.37763,0) ]
 [ (0.123458,0.318933) ]
 [ (-0.0764131,0.0695855) ]


=======================
Clebsch-Gordan products
=======================

The ``CGproduct`` function computes a single part (indexed by the last argument) 
of the part of Clebsch--Gordan product of two SO3-parts. 

.. code-block:: python

 >>> A=gelib.SO3part.gaussian(2,2)
 >>> B=gelib.SO3part.gaussian(2,2)
 >>> C=gelib.CGproduct(A,B,3)
 >>> print(C)
 [ (-1.99979,-0.121461) (-3.58782,2.10019) (-1.34679,-1.51318) (-2.83808,-0.352546) ]
 [ (1.14425,0.901388) (2.9222,0.910338) (-0.409205,0.741729) (-1.40359,1.8079) ]
 [ (-0.19909,-0.168839) (-3.0053,0.370446) (-0.0255721,0.566075) (-1.04462,-1.07568) ]
 [ (-0.149099,2.08319) (1.11618,0.282956) (-1.0153,0.660029) (1.42106,0.50812) ]
 [ (0.474459,-1.64466) (0.234653,0.618942) (0.828522,-0.762771) (-0.895682,0.300558) ]
 [ (-0.995215,0.783116) (0.885221,-0.726837) (-0.404905,-0.579419) (0.149155,0.764922) ]
 [ (-0.387969,-1.5089) (-0.163056,1.1043) (0.628268,-0.465748) (-0.576328,0.309953) ]
 
The Clebsch-Gordan product is an equivariant operation. This can be verified as follows:

.. code-block:: python

  >>> A=gelib.SO3part.gaussian(2,2)
  >>> r=gelib.SO3element.uniform()
  >>>
  >>> B=gelib.CGproduct(A,A,2)
  >>> Br=B.rotate(r)
  >>> print(Br)
  [ (-2.29009,1.34611) (1.87151,1.91239) (1.87151,1.91239) (1.39403,-1.47615) ]
  [ (-2.4749,0.63966) (-2.28298,1.25885) (-2.28298,1.25885) (-2.25532,0.710633) ]
  [ (-0.46347,0.938204) (1.08875,0.235826) (1.08875,0.235826) (0.633137,2.93817) ]
  [ (1.75711,0.22648) (-1.34673,0.988074) (-1.34673,0.988074) (2.81846,-0.208569) ]
  [ (-3.34617,1.96189) (-0.792441,-1.00736) (-0.792441,-1.00736) (-1.36723,-0.665336) ]
  >>>
  >>> Ar=A.apply(r)
  >>> Br2=gelib.CGproduct(Ar,Ar,2)  # by equiariance should be the same as Br
  >>> print(Br2)
  [ (-2.29009,1.34611) (1.87151,1.91239) (1.87151,1.91239) (1.39403,-1.47615) ]
  [ (-2.4749,0.639659) (-2.28298,1.25885) (-2.28298,1.25885) (-2.25532,0.710633) ]
  [ (-0.46347,0.938204) (1.08875,0.235827) (1.08875,0.235826) (0.633138,2.93817) ]
  [ (1.75711,0.22648) (-1.34673,0.988074) (-1.34673,0.988074) (2.81846,-0.20857) ]
  [ (-3.34617,1.96189) (-0.792441,-1.00736) (-0.792441,-1.00736) (-1.36723,-0.665336) ]


==============
GPU operations
==============


Similarly to ``cnine`` tensors, if `GElib` was compiled with GPU support and a GPU is avaliable, 
``SO3part`` objects can be moved back and forth between the host (CPU) and the GPU. 
In general, if the operands are on the host, any given operation will be performed on the host and 
the result is placed on the host. Conversely, if the operands are on the GPU, 
the operation will be performed on the GPU and the result will be placed on the same GPU. 

.. code-block:: python

  >>> A=gelib.SO3part.gaussian(4,4)
  >>> B=A.to(1) # Create a copy of A on the first GPU (GPU0)
  >>> B.device() # Return whether the SO3part is resident on the CPU or GPU
  1   
  >>> C=B.to(0) # Move B back to the host 
  >>> B.device() 
  0

