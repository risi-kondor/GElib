******
SO3vec
******


An ``SO3vec`` object represents a general SO(3)-covariant vector. 
On the backend,  ``SO3vec`` is stored as a collection 
of ``SO3part`` objects with :math:`\ell=0,1,2,\ldots,L`. 

An ``SO3vec`` can be initialized from an ``SO3type`` object or just a vector of multiplicities 
corresponding to :math:`\ell=0,1,2,\ldots,L`.  

.. code-block:: python

 >>> tau=SO3type([2,3,1])
 >>> v=SO3vec.gaussian(tau)
 >>> v=SO3vec.gaussian([2,3,1])
 >>> v
 <GElib::SO3vec of type(2,3,1)>
 >>> print(v)
 Part l=0:
   [ (-1.50279,-0.929941) (0.570759,-0.934988) ]

 Part l=1:
   [ (-0.764676,0.537104) (0.250854,0.694816) (-0.188164,0.541231) ]
   [ (-1.51315,-1.96886) (1.32256,0.354178) (1.93468,-0.455262) ]
   [ (1.25244,0.377994) (1.0417,-0.274068) (-0.696964,0.005616) ]

 Part l=2:
   [ (-1.77286,-1.83641) ]
   [ (0.519691,-0.257851) ]
   [ (0.0431933,-0.391737) ]
   [ (-1.96668,2.69588) ]
   [ (-0.480737,1.6585) ]


The type of an SO(3)-vector and the total number of parts can be accessed as follows.

.. code-block:: python

 >>> v.type()
 <GElib::SO3type(2,3,1)>
 >>> len(v)
 3


==============================
Accessing parts and arithmetic
==============================

Individual parts of the vector can be accessed as follows.

.. code-block:: python

 >>> A=v[1]
 >>> print(A)
 [ (1.3828,-0.89237) (0.0523187,-0.228782) (-0.904146,1.16493) ]
 [ (1.87065,0.584898) (-1.66043,-0.660558) (-0.688081,0.534755) ]
 [ (0.0757219,-0.607787) (1.47339,0.74589) (0.097221,-1.75177) ]

.. code-block:: python

 >>> v[1]=SO3part.ones(1,2)
 >>> print(v)
 Part l=0:
   [ (-0.0893479,0.200541) (2.04214,0.277807) ]

 Part l=1:
   [ (1,0) (1,0) ]
   [ (1,0) (1,0) ]
   [ (1,0) (1,0) ]

 Part l=2:
   [ (1.34974,0.628626) ]
   [ (1.40554,1.48798) ]
   [ (1.14327,1.28343) ]
   [ (0.862926,1.76615) ]
   [ (-0.409665,1.25511) ]


Arithmetic operations can be applied to ``SO3vec`` objects just as to ``SO3part`` s.

.. code-block:: python

 >>> v=SO3vec.gaussian([2,3,1])
 >>> u=SO3vec.gaussian([2,3,1])
 >>> w=u+2*v
 >>> print(w)
 Part l=0:
   [ (0.432785,-1.26372) (-0.904954,4.84177) ]

 Part l=1:
   [ (0.648943,4.03096) (-0.83821,2.60792) (-2.37467,-2.41012) ]
   [ (-4.0278,-2.6327) (1.05469,0.0868182) (-2.66474,0.674749) ]
   [ (1.0687,0.392436) (-3.35649,-3.66103) (-2.18214,0.830834) ]
 
 Part l=2:
   [ (-1.06816,2.30383) ]
   [ (-4.85571,-1.90676) ]
   [ (-1.57795,1.02786) ]
   [ (0.58204,-0.316313) ]
   [ (0.097331,0.975934) ]


===========================
Functions of SO3vec objects
===========================

Arithmetic operations can be applied to ``SO3vec`` objects just as to ``SO3part`` s.

.. code-block:: python

 >>> u=SO3vec.gaussian([1,1])
 >>> v=SO3vec.gaussian([1,1])
 >>> inp(u,v)
 (5.523734092712402-1.4036915302276611j)
 >>> norm2(u)
 (10.693071365356445+0j)


=======================
Clebsch-Gordan products
=======================

The Clebsch-Gordan product of two SO3-vectors can be computed as follows.

.. code-block:: python

 >>> u=SO3vec.gaussian([1,1])
 >>> v=SO3vec.gaussian([1,1])
 >>> w=CGproduct(u,v)
 >>> print(w)
 Part l=0:
   [ (0.800454,-2.72231) (0.387997,-2.21325) ]
 
 Part l=1:
   [ (-1.08378,0.166964) (-1.13947,1.02458) (-0.979756,-0.170846) ]
   [ (-3.14667,-0.020229) (-1.60544,-0.595765) (-0.658927,1.13758) ]
   [ (0.573493,3.50629) (0.609701,-0.290724) (-1.86063,-0.256204) ]
 
 Part l=2:
   [ (0.545523,0.23039) ]
   [ (1.0578,1.10345) ]
   [ (0.098245,0.754121) ]
   [ (1.15855,-0.537074) ]
   [ (-0.530323,0.658823) ]




==============
GPU operations
==============


Similarly to ``cnine`` tensors, ``SO3part`` objects, ``SO3vec`` objects 
 can moved back and forth between the host (CPU) and the GPU with the ``to`` method. 

.. code-block:: python

  >>> A=SO3vec.gaussian([2,3,1])
  >>> B=A.to(1) # Create a copy of A on the first GPU (GPU0)
  >>> C=B.to(0) # Move B back to the host 


