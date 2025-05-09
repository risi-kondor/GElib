***********
SO3vecArr
***********

``SO3vecArr`` is the arrayed counterpart of the ``SO3vec`` class, stored as a 
list of ``SO3partArr`` s. 

.. code-block:: python

 >> v=gelib.SO3vecArr.randn(1,[2,2],{0:2,1:3,2:1})
 >> print(v.repr())

 <GElib::SO3vecB_array of type(1,(2,2),(2,3,1))>

 >> print(v)

 Part l=0:
   Cell(0,0,0)
   [ (-1.67126,-0.419833) (-2.07704,-0.409032) ]

   Cell(0,0,1)
   [ (0.399491,0.698713) (-1.36641,0.138502) ]

   Cell(0,1,0)
   [ (0.296908,-0.669854) (0.873029,-0.26619) ]

   Cell(0,1,1)
   [ (1.39529,-0.0214446) (0.0659465,-0.840586) ]


 Part l=1:
   Cell(0,0,0)
   [ (1.23563,0.231609) (-0.706519,-0.289283) (-0.175813,0.139454) ]
   [ (1.07154,-0.0482395) (0.201686,0.238003) (2.18152,-1.02785) ]
   [ (-0.204963,-0.643517) (1.13979,-1.28142) (-1.97073,-0.0872069) ]

   Cell(0,0,1)
   [ (-1.41811,-0.385749) (0.845764,0.314845) (0.157016,0.813198) ]
   [ (-0.0473072,0.489402) (-0.48208,-0.175419) (-0.713555,-1.36876) ]
   [ (-1.43885,0.282084) (0.548364,0.765744) (-1.33589,-0.124562) ]

   Cell(0,1,0)
   [ (-0.416213,-2.10858) (-0.436183,-1.95774) (-0.4028,0.206586) ]
   [ (0.605511,-1.35408) (-1.04419,1.03389) (-0.443257,-1.82339) ]
   [ (1.78785,-1.41791) (-1.03278,0.820731) (0.38561,0.0195039) ]

   Cell(0,1,1)
   [ (0.487334,-0.34982) (-0.655663,1.94033) (-0.246383,-1.03738) ]
   [ (0.61681,-0.212469) (1.16121,1.27737) (0.658922,-2.23992) ]
   [ (-0.379091,-0.174235) (-0.430056,0.523456) (0.657717,0.141875) ]


 Part l=2:
   Cell(0,0,0)
   [ (1.40848,-0.838271) ]
   [ (2.1893,-0.384493) ]
   [ (-1.21339,-0.393357) ]
   [ (1.87515,0.716365) ]
   [ (-0.0917885,-0.0592561) ]

   Cell(0,0,1)
   [ (-0.599226,0.901921) ]
   [ (-0.657987,1.73258) ]
   [ (0.95926,1.65543) ]
   [ (1.75915,-1.78865) ]
   [ (0.534479,-1.64673) ]

   Cell(0,1,0)
   [ (-0.497837,-0.619745) ]
   [ (-0.0603971,-0.613755) ]
   [ (-0.181712,-0.923199) ]
   [ (-1.34202,0.259467) ]
   [ (-1.03544,-1.12516) ]

   Cell(0,1,1)
   [ (1.7281,0.0535397) ]
   [ (-0.287769,-0.284569) ]
   [ (-1.21629,0.981768) ]
   [ (-0.234547,-0.367627) ]
   [ (-0.190083,0.268198) ]


Similarly to ``SO3partArr``, ``SO3vecArr`` supports the same operations as ``SO3vec``, plus some efficient 
reductions and broadcast operations.  

======
Gather
======

``SO3vecArr`` supports ``gather`` functionality, just like ``SO3partArr``:

.. code-block:: python

 >> import cnine
 >> C=torch.tensor([[0,1,0],[0,0,0],[1,0,0]],dtype=torch.float32);
 >> mask=cnine.Rmask1(C)
 >> X=gelib.SO3vecArr.randn(1,[3],[1,1,1])
 >> Y=X.gather(mask)

