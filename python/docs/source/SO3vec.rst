******
SO3vec
******


An ``SO3vec`` object represents a general SO(3)-covariant vector and is stored as a collection 
of ``SO3part`` objects with :math:`\ell=0,1,2,\ldots,L`. 

An ``SO3vec`` is initialized from an ``SO3type`` object or just a vector of multiplicities 
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

===============
Accessing parts
===============

Individual parts of the vector can be extracted as follows.

.. code-block:: python

 >>> A=v[1]
 >>> print(A)
 [ (1.3828,-0.89237) (0.0523187,-0.228782) (-0.904146,1.16493) ]
 [ (1.87065,0.584898) (-1.66043,-0.660558) (-0.688081,0.534755) ]
 [ (0.0757219,-0.607787) (1.47339,0.74589) (0.097221,-1.75177) ]


==========
Arithmetic
==========


==============
GPU operations
==============


Similarly to ``cnine`` tensors, ``SO3part`` objects, ``SO3vec`` objects 
 can moved back and forth between the host (CPU) and the GPU with the ``to`` method. 

.. code-block:: python

  >>> A=SO3vec.gaussian([2,3,1])
  >>> B=A.to(1) # Create a copy of A on the first GPU (GPU0)
  >>> C=B.to(0) # Move B back to the host 


