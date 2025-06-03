*************
Tensor arrays
*************

The basic device in `cnine` for parallelizing computations on the GPU are tensor arrays. 
A tensor array is an array of tensors of the same size arranged in a contiguous data structure.  
An ``rtensorArr`` object is an array of real tensors, whereas an ``ctensorArr`` object is 
an array of complex tensors.  

We describe the basic functionality of tensor arrays for the ``rtensorArr`` class. 
The API of ``ctensorArr`` is analogous apart from the addition of some functions taking complex conjugates, 
etc.. 


======================
Creating tensor arrays
======================


The following example shows how to create a :math:`2\times 2` array of :math:`4\times 4` 
dimensional tensors filled with random numbers.

.. code-block:: python

  >>> A=cnine.rtensorArr.gaussian([2,2],[4,4])
  >>> print(A)
  Cell (0,0)
  [ -1.23974 -0.407472 1.61201 0.399771 ]
  [ 1.3828 0.0523187 -0.904146 1.87065 ]
  [ -1.66043 -0.688081 0.0757219 1.47339 ]
  [ 0.097221 -0.89237 -0.228782 1.16493 ]


  Cell (0,1)
  [ -1.50279 0.570759 -0.929941 -0.934988 ]
  [ -0.764676 0.250854 -0.188164 -1.51315 ]
  [ 1.32256 1.93468 1.25244 1.0417 ]
  [ -0.696964 0.537104 0.694816 0.541231 ]


  Cell (1,0)
  [ -1.13769 -1.22027 0.111152 -0.672931 ]
  [ -1.39814 -0.477463 0.643125 1.37519 ]
  [ -1.2589 0.259477 -1.6247 -0.996947 ]
  [ -0.149277 -1.3338 -1.44352 0.65806 ]


  Cell (1,1)
  [ -1.20183 -0.399454 -0.727057 0.43853 ]
  [ -0.42954 -2.20967 -1.22569 0.73464 ]
  [ 0.630166 0.137796 0.674001 -0.281158 ]
  [ -1.1945 1.06918 -1.2115 -2.1947 ]
 

=======================
Tensor array dimensions
=======================

The array dimensions and cell dimensions of a tensor array are accessed as follows.

.. code-block:: python

  >>> A=cnine.rtensorArr.gaussian([2,2],[4,4])
  >>> adims=A.get_adims()
  >>> print(adims)
  (2,2)
  >>> cdims=A.get_cdims()
  >>> print(cdims)
  (4,4)


=========================
Accessing tensor cells
=========================


Individual tensors in the tensor array, called `cells`, are accessed similarly to how tensor 
elements are accessed in regular tensors. 

.. code-block:: python

  >>> B=A([0,1])
  >>> print(B)
  [ -1.50279 0.570759 -0.929941 -0.934988 ]
  [ -0.764676 0.250854 -0.188164 -1.51315 ]
  [ 1.32256 1.93468 1.25244 1.0417 ]
  [ -0.696964 0.537104 0.694816 0.541231 ]


.. code-block:: python

  >>> A[[0,1]]=A([0,0])
  >>> print(A)
  Cell (0,0)
  [ -1.23974 -0.407472 1.61201 0.399771 ]
  [ 1.3828 0.0523187 -0.904146 1.87065 ]
  [ -1.66043 -0.688081 0.0757219 1.47339 ]
  [ 0.097221 -0.89237 -0.228782 1.16493 ]


  Cell (0,1)
  [ -1.23974 -0.407472 1.61201 0.399771 ]
  [ 1.3828 0.0523187 -0.904146 1.87065 ]
  [ -1.66043 -0.688081 0.0757219 1.47339 ]
  [ 0.097221 -0.89237 -0.228782 1.16493 ]


  Cell (1,0)
  [ -1.13769 -1.22027 0.111152 -0.672931 ]
  [ -1.39814 -0.477463 0.643125 1.37519 ]
  [ -1.2589 0.259477 -1.6247 -0.996947 ]
  [ -0.149277 -1.3338 -1.44352 0.65806 ]


  Cell (1,1)
  [ -1.20183 -0.399454 -0.727057 0.43853 ]
  [ -0.42954 -2.20967 -1.22569 0.73464 ]
  [ 0.630166 0.137796 0.674001 -0.281158 ]
  [ -1.1945 1.06918 -1.2115 -2.1947 ]


==================================
Conversion to/from PyTorch tensors
==================================

A tensors array with :math:`d` cell dimensions and :math:`D` array dimensions can be converted to 
a ``torch.tensor`` with :math:`D+d` dimensions.

.. code-block:: python

  >>> A=cnine.rtensorArr.sequential([2,2],[3,3])
  >>> print(A)
  Cell (0,0)
  [ 0 1 2 ]
  [ 3 4 5 ]
  [ 6 7 8 ]

  Cell (0,1)
  [ 9 10 11 ]
  [ 12 13 14 ]
  [ 15 16 17 ]

  Cell (1,0)
  [ 18 19 20 ]
  [ 21 22 23 ]
  [ 24 25 26 ]

  Cell (1,1)
  [ 27 28 29 ]
  [ 30 31 32 ]
  [ 33 34 35 ]

  >>> B=A.torch()
  >>> B
  tensor([[[[ 0.,  1.,  2.],
            [ 3.,  4.,  5.],
            [ 6.,  7.,  8.]],
  
           [[ 9., 10., 11.],
            [12., 13., 14.],
            [15., 16., 17.]]],
  
  
          [[[18., 19., 20.],
            [21., 22., 23.],
            [24., 25., 26.]],
  
           [[27., 28., 29.],
            [30., 31., 32.],
            [33., 34., 35.]]]])

Conversely, a ``torch.tensor`` can be converted into a ``rtensorArr`` but we need to specify how many of its 
leading dimensions are to be interpreted as array dimensions. 

.. code-block:: python

  >>> A=torch.rand([2,3,3])
  >>> A
  tensor([[[0.3004, 0.4147, 0.5666],
          [0.7969, 0.2912, 0.8442],
           [0.9161, 0.7182, 0.4490]],

          [[0.5466, 0.3649, 0.1898],
           [0.5851, 0.2558, 0.2237],
           [0.8992, 0.7448, 0.0836]]])
  >>> B=cnine.rtensorArr(1,A)
  >>> print(B)
  Cell (0)
  [ 0.30044 0.414732 0.566644 ]
  [ 0.796893 0.291165 0.844217 ]
  [ 0.916076 0.718188 0.449004 ]

  Cell (1)
  [ 0.546558 0.36489 0.189827 ]
  [ 0.585105 0.255816 0.223677 ]
  [ 0.899194 0.744844 0.083603 ]


Complex tensor arrays are converted similarly, with the resulting ``torch.tensor`` acquiring an extra 
leading dimension of size two corresponding to the real and imaginary parts. 

=====================
Cellwise operations 
=====================

Tensor arrays support the same arithmetic operations as regular tensors. 
By default, a given operation is applied to each cell of the array independently. 
For example, the result of adding an :math:`n\times m` tensor array ``A`` to another :math:`n\times m` 
tensor array ``A`` is an :math:`n\times m` array in which the :math:`(i,j)` cell is the sum 
of the corresponding cells in ``A`` and ``B``.

.. code-block:: python

  >>> A=cnine.rtensorArr.zero([2,2],[3,3])
  >>> B=cnine.rtensorArr.ones([2,2],[3,3])
  >>> C=A+B
  >>> print(C([0,1]))
  [ 1 1 1 ]
  [ 1 1 1 ]
  [ 1 1 1 ]


====================
Broadcast operations
====================

Applying a binary operation to a tensor array and a regular tensor corresponds to 
first broadcasting the tensor to an array of the same size and then applying the operation cellwise.

.. code-block:: python

 >>> A=cnine.rtensorArr.zero([2,2],[3,3])
 >>> B=cnine.rtensor.ones([3,3])
 >>> C=A+B
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


======================
Widening and reduction
======================

Summing the array along a given array dimension is called `reduction`, whereas copying it multiple times to 
create a new array dimension is called `widening`. 
On the GPU, both these operations are performed in `cnine` with fast, parallelized algorithms. 

.. code-block:: python

  >>> A=cnine.rtensorArr.gaussian([2,2],[4,4])
  >>> B=A.reduce(1)
  >>> print(B)
  Cell (0)
  [ -0.610066 -1.75872 0.0605343 0.221048 ]
  [ -0.485987 0.911379 -0.117453 -2.9732 ]
  [ -2.15961 1.34379 0.878445 0.246828 ]
  [ -0.993059 -0.996571 0.578766 -1.27511 ]


  Cell (1)
  [ 1.6495 -1.15005 2.06733 -1.53783 ]
  [ -1.38929 0.878757 0.348551 0.871658 ]
  [ -2.09839 -0.0545999 -1.23761 0.399476 ]
  [ -1.30456 -0.378178 1.31794 0.917212 ]
  
  
  >>> C=B.widen(1,3)
  >>> print(C)
  Cell (0,0)
  [ -0.610066 -1.75872 0.0605343 0.221048 ]
  [ -0.485987 0.911379 -0.117453 -2.9732 ]
  [ -2.15961 1.34379 0.878445 0.246828 ]
  [ -0.993059 -0.996571 0.578766 -1.27511 ]


  Cell (0,1)
  [ -0.610066 -1.75872 0.0605343 0.221048 ]
  [ -0.485987 0.911379 -0.117453 -2.9732 ]
  [ -2.15961 1.34379 0.878445 0.246828 ]
  [ -0.993059 -0.996571 0.578766 -1.27511 ]


  Cell (0,2)
  [ -0.610066 -1.75872 0.0605343 0.221048 ]
  [ -0.485987 0.911379 -0.117453 -2.9732 ]
  [ -2.15961 1.34379 0.878445 0.246828 ]
  [ -0.993059 -0.996571 0.578766 -1.27511 ]
  
  
  Cell (1,0)
  [ 1.6495 -1.15005 2.06733 -1.53783 ]
  [ -1.38929 0.878757 0.348551 0.871658 ]
  [ -2.09839 -0.0545999 -1.23761 0.399476 ]
  [ -1.30456 -0.378178 1.31794 0.917212 ]


  Cell (1,1)
  [ 1.6495 -1.15005 2.06733 -1.53783 ]
  [ -1.38929 0.878757 0.348551 0.871658 ]
  [ -2.09839 -0.0545999 -1.23761 0.399476 ]
  [ -1.30456 -0.378178 1.31794 0.917212 ]


  Cell (1,2)
  [ 1.6495 -1.15005 2.06733 -1.53783 ]
  [ -1.38929 0.878757 0.348551 0.871658 ]
  [ -2.09839 -0.0545999 -1.23761 0.399476 ]
  [ -1.30456 -0.378178 1.31794 0.917212 ]


=================
GPU functionality
=================

Tensor arrays can moved back and forth between the host (CPU) and the GPU similarly to tensors. 

.. code-block:: python

  >>> A=cnine.rtensorArr.sequential([2,2],[4,4],device=1) # create a tensor array on the GPU
  >>> A.device() # print out where A is stored
  1
  >>> B=A.to(0) # Create a copy of A on the CPU
  >>> B.device() # print out where B is stored 
  0



..note::
 The default C++ backend class for real tensors arrays is ``RtensorArrayA`` 
 and for complex tensor arrays is ``CtensorArrayA``. 
 ``RtensorArrayA`` stores an :math:`D_1\times \ldots \times D_K` array of  :math:`d_1\times\ldots\times d_k` 
 dimensional tensors in a similar way that ``RtensorA`` would store a single  
 :math:`D_1\times \ldots \times D_K\times d_1\times\ldots\times d_k` dimensional tensor. 
 An important caveat, however, is that 
 the stride between consecutive cells is rounded up to the nearest multiple of 128 bytes. 
 While this facilitates memory access, especially on the GPU, it makes it somewhat harder to 
 convert a ``tensor_array`` object to a single tensor in e.g.. `PyTorch`. 
 The ``CtensorArrayA`` class stores a complex tensor array in a single array, 
 consisting of two real tensor arrays back to back. 

 A tensor array object's header, including information about tensor dimensions, strides, etc., is always resident on 
 the host. When a tensor array is moved to the GPU, only the array containing the actual tensor entries 
 is moved to the  GPU's global memory. 

