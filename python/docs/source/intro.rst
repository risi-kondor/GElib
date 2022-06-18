############
Introduction
############

SO(3) is the group of three dimensional rotations. 
SO(3)-equivariant neural networks learn functions that behave in specific ways with respect to this group.  
Such architectures have recently become popular in a range of applications  
in physics, chemistry, computational biology and computer vision.

From a mathematical point of view, the key feature of these architectures is that  
the output of each neuron is an SO(3)-covariant quantity, which means that under rotations  
it transforms either according to a single irreducible representation (irrep) of SO(3) or a specific 
combination of irreducible representations. This property is essential for ensuring equivariance,  
but also puts restrictions on what operations are permissible in the network. 

.. 
 For example, a given application might demand that the overall output be rotation-invariant or 
 transform the exact same way as the inputs transform. 

GElib provides two basic classes to facilitate implementing SO(3)-equivariant networks:

#. The ``SO3part`` class stores any quantity that transforms according to a single irrep. 
#. The ``SO3vec`` class stores a quantity that transforms according to a combination of different irreps.

The library also implements the most important operations that can be performed on SO(3)-covariant objects, 
specifically, different variants of the Clebsch--Gordan product. 

********
Features
********

GElib provides the following features:

#. Classes to store and manipulate SO(3)-covariant (equivariant) vectors.
#. Fast implementations of different variants of the Clebsch-Gordan transforms on both the CPU and the GPU.
#. Facilities for operating on arrays of SO(3)-covariant vectors in parallel, 
   including in irregular patterns (graphs).
#. Interoperability with PyTorch's ``autograd`` functionality. 

 
************
Installation
************

GElib is installed as a PyTorch C++ (or CUDA) extension and requires the following: 

#. C++14 or higher
#. PyTorch
#. cnine (see below) 

GElib is easiest to install with ``pip``:

#. Download the `cnine <https://github.com/risi-kondor/cnine>`_  library 
   and install it on your system
#. Download `GElib <https://github.com/risi-kondor/GElib>`_. 
   By default, it is assumed that cnine and GElib are downloaded to the same directory 
   (e.g., ``Downloads``).      
#. Edit the user configurable variables in ``python/setup.py`` as necessary. 
#. Run the command ``pip install -e .`` in the ``GElib/python`` directory. 

..
   #. Run ``python setup.py install`` in the ``python`` directory to compile the package and install it on your system.
    cnine does not need to be separately installed on your system, but the 
   cnine source files are required for the GElib intallation process itself. 

*************
Configuration
*************

The installation can be configured by setting the following variables in ``python/setup.py``.

``compile_with_cuda``
  If set to ``True``, `GElib` will be compiled with GPU suport. This requires a working CUDA and CUBLAS installation 
  on your system and PyTorch itself having been compiled with CUDA enabled. If `GElib` is compiled with CUDA,  
  you must always import ``torch`` before importing ``GElib``.

``copy_warnings``
  If set to ``True``, `GElib` will print a message to the terminal whenever a data object 
  is copied or move-copied. This option is useful for code optimization. 

``torch_convert_warnings`` 
  If set to ``True``, `cnine` will print a message to the terminal whenver a data object is explicitly 
  converted (as opposed to just forming a tensor view) to/from PyTorch format. 



***************
Troubleshooting
***************

.. 
   #. If it becomes necessary to change the location where `setuptools` 
     places the compiled module, add a file called ``setup.cfg`` 
     with content 

      .. code-block:: none
   
      [install]
      prefix=<target directory where you want the module to be placed>

   in the ``python`` directory. Make sure that the new target directory is in Python's load path.

#. PyTorch C++ extensions must be compiled against the same version of CUDA that PyTorch   
   itself was compiled with. If this becomes an issue, it might be necessary to install an 
   alternative version of CUDA on your system and force `setuptools` to use that version by setting 
   the ``CUDA_HOME`` enironment variable, as, e.g. 

   .. code-block:: none
   
    export CUDA_HOME=/usr/local/cuda-11.3


************
Known issues
************



 
*****
Usage 
*****

To load the ``gelib`` module in Python, use the command ``import gelib``. 
Depending on whether or not GElib was installed with CUDA, the message 
``Starting GElib with GPU support`` or ``Starting GElib without GPU support`` will appear, 
confirming that `gelib` has successfully started and initialized its 
static data objects. 

..
  GElib has two distinct interfaces implemented in two different modules:

  #. To use the library *without* PyTorch's autodiff functionality, load the library with ``import gelib_base as gelib``. 
  #. To use the library *with* automatic differentiation, load the library with ``import gelib_torch as gelib``. 

  The two modules use identical syntax, therefore the following documentation applies to both. 
  The backend implementation of the two modules however is quite different. 
  ``gelib_base`` is just a wrapper for the underlying C++ classes. 
  In contrast, for interoperability with ``torch.autodiff``, 
  ``gelib_torch`` 's core classes are Python classes derived from ``torch.tensor``. 
  These Python classes, in turn, call the wrappers implemented in ``gelib_base``.  
  Inevitably, this incurs some performance overhead.  

.. 
  Since ``gelib_torch`` is built on ``gelib_base``, the two modules can also be used together.   


