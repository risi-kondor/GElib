****************************
Cell maps and cell operators
****************************

`cnine` takes advantage of the parallelism afforded by GPUs via cell maps and cell operators. 
A `cell operator` is an operation such as tensor addition, matrix multiplication, etc., that can 
be applied to a single cell in a tensor array or a pair of cells in two different tensor 
arrays, yielding a single cell as a result. A cell operator typically defines two different 
implementations of the same operation: one for the case when the operands are on the host, and one 
for the case when they are on the GPU.

A `cell map` is a C++ template class that determines in what pattern a given cell operator is 
applied to a combination of tensor arrays. For example, the cell map can be cellwise, capturing 
an outer product, inner product or determined by a cell mask object corresponding to a graph of 
interactions.

The power of the cell map/operator formalism lies in the fact that any cell operator can be combined 
with any cell map. Moreover, when applied on the GPU, `cnine` will perform the individual cell operations 
in parallel at the thread block level, meaning that each cell operation will be mapped to a separate 
streaming multiprocessor. Thus, according to the general architectural restrictions of NVIDIA GPU's, 
the cell operation itself can utilize up to 1024 GPU threads, which can efficiently share data 
with each other via the multiprocessors shared memory. 

The use of cell maps and cell operators from Python is limited by the fact that on the C++ 
side they are implemented via templates. It is not feasible to compile all possible template 
instantiations into the Python ``cnine`` module. Rather, for any new combination of cell operator/
cell map, the Python binding code in ``cnine_py.cpp`` has to be updated manually and 
the module must be recompiled. 

Here we demonstrate the cell map functionality via a single cell operator, ``RtensorA_add_plus_cop``.

=============
Cellwise Cmap
=============


.. code-block:: python

 >>> A=rtensor_arr.zero([2,2],[3,3])
 >>> B=rtensor_arr.ones([2,2],[3,3])
 >>> C=rtensor_arr.zero([2,2],[3,3])
 >>> cellwise_add_plus(C,A,B)
 >>> print(C)
 Cell (0,0)
 [ 1 1 1 ]
 [ 1 1 1 ]
 [ 1 1 1 ]


 Cell (0,1)
 [ 1 1 1 ]
 [ 1 1 1 ]
 [ 1 1 1 ]


 Cell (1,0)
 [ 1 1 1 ]
 [ 1 1 1 ]
 [ 1 1 1 ]


 Cell (1,1)
 [ 1 1 1 ]
 [ 1 1 1 ]
 [ 1 1 1 ]


==========
Inner Cmap
==========


.. code-block:: python

 >>> A=rtensor_arr.zero([2],[3,3])
 >>> B=rtensor_arr.ones([2],[3,3])
 >>> C=rtensor_arr.zero([1,1],[3,3])
 >>> inner_add_plus(C,A,B)
 >>> print(C)
 Cell (0,0)
 [ 2 2 2 ]
 [ 2 2 2 ]
 [ 2 2 2 ]


==========
Outer Cmap
==========


.. code-block:: python

 >>> A=rtensor_arr.zero([2],[3,3])
 >>> B=rtensor_arr.ones([2],[3,3])
 >>> C=rtensor_arr.zero([2,2],[3,3])
 >>> outer_add_plus(C,A,B)
 >>> print(C)
 Cell (0,0)
 [ 1 1 1 ]
 [ 1 1 1 ]
 [ 1 1 1 ]


 Cell (0,1)
 [ 1 1 1 ]
 [ 1 1 1 ]
 [ 1 1 1 ]


 Cell (1,0)
 [ 1 1 1 ]
 [ 1 1 1 ]
 [ 1 1 1 ]


 Cell (1,1)
 [ 1 1 1 ]
 [ 1 1 1 ]
 [ 1 1 1 ]

==========
Mprod cmap
==========


.. code-block:: python

 >>> A=rtensor_arr.ones([2,2],[3,3])
 >>> B=rtensor_arr.ones([2],[3,3])
 >>> C=rtensor_arr.zero([2],[3,3])
 >>> mprod_add_plus(C,A,B)
 >>> print(C)
 Cell (0)
 [ 4 4 4 ]
 [ 4 4 4 ]
 [ 4 4 4 ]


 Cell (1)
 [ 4 4 4 ]
 [ 4 4 4 ]
 [ 4 4 4 ]

