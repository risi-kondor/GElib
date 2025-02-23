***********
SO3 weights
***********

It can be shown that the most general `linear` equivariant operation for ``SO3vec`` objects 
is to linearly mix the fragments belonging to the same ``SO3part`` amongst themselves.  
In matrix language, this corresponds to multiplying each constituent ``SO3part`` by a 
complex matrix of the appropriate dimensions from the right. 
The ``SO3weights`` class is a container for a sequence of such matrices corresponding to each part.

A zero or random ``SO3weights`` object ``W`` can be constructed by specifying the row and column dimensions 
of the matrices, i.e., the input and output type of the transformation :math:`u\mapsto v\cdot W`.
For example, the following intiializes an ``SO3weights``  object that can transform a vector of 
type (1,2,2) to a vector of type (3,2,1).

.. code-block:: python

 >>> w=gelib.SO3weights.randn([1,2,2],[3,2,1])
 >>> print(w)
 tensor 0:
   [ (-0.460932,-0.24965) (1.04747,-0.113222) (-1.19098,-0.230981) ]

 tensor 1:
   [ (1.00428,-0.897704) (-1.37185,1.08807) ]
   [ (-1.81899,-1.16288) (0.156111,0.516122) ]

 tensor 2:
   [ (0.635841,0.17247) ]
   [ (1.44829,0.592355) ]


If we define an ``SO3vec`` of the appropriate type, ``w`` can be directly applied to it. 
Naturally, the same weight matrix is applied to each slice of ``u`` corresponding 
to different values of the batch index. 

.. code-block:: python

 >>> u=gelib.SO3vec.randn(1,[1,2,2])
 >>> v=u*w
 >>> print(v)
 Part l=0:
   [ (1.0454,-0.168498) (-1.53694,1.47214) (2.19084,-1.09811) ]


 Part l=1:
   [ (-2.23171,0.0901575) (0.648155,1.02873) ]
   [ (-1.88365,2.93436) (1.65895,-0.630005) ]
   [ (1.1317,-1.46948) (-1.4877,0.956643) ]


 Part l=2:
   [ (0.852567,-3.69071) (1.38565,-2.07584) (-0.87405,1.31016) (1.69347,1.64524) (-1.23533,2.00124) ]

The ``SO3weights`` class also supports some basic arithmetic operations such as addition, multiplication 
by scalars, and so on.

.. code-block:: python

 >>> w2=w*torch.tensor([2.])
 >>> print(w2)
 tensor 0:
   [ (-1.47271,0.681177) (0.377892,0.0660075) (1.51043,2.94915) ]

 tensor 1:
   [ (2.48892,-0.946779) (0.433693,-0.771083) ]
   [ (-2.26782,2.61198) (-1.01805,-4.51223) ]

 tensor 2:
   [ (0.968962,-2.40745) ]
   [ (-0.650463,0.465023) ]


  
