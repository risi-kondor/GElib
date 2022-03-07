GElib is a C++ library with a CUDA backend for computations related to 
the rotation group SO(3), specifically for building SO(3)-equivariant 
neural networks. 
GElib is developed by Risi Kondor at the University of Chicago. 
GElib is released under the 
`Mozilla public license v.2.0 <https://www.mozilla.org/en-US/MPL/2.0/>`_ 
and can be downloaded from `https://github.com/risi-kondor/GElib <https://github.com/risi-kondor/GElib>`_.

This document provides documentation for GElib's PyTorch interface. 
The PyTorch interface only provides access to a subset of the C++ library's functionality. 

GElib is currently in alpha. Some features described below may still be incomplete or untested. 

########
Features
########

#. Classes to store and manipulate SO(3)-equivariant vectors.
#. Fast implementation of the Clebsch-Gordan transform for both the CPU and the GPU.
#. Facilities for operating on arrays of SO(3)-equivariant vectors in parallel, 
   including in irregular patterns (graphs).
#. Interoperability with PyTorch's ``autograd`` functionality. 

 
############
Installation
############

GElib is installed as a PyTorch C++ (or CUDA) extension and requires the following: 

#. C++14 or higher
#. PyTorch
#. cnine (see below) 

To install GElib follow these steps:

#. Download the `cnine <https://github.com/risi-kondor/cnine>`_  library and install it on your system. 
#. Download `GElib <https://github.com/risi-kondor/GElib>`_. 
#. Edit the user configurable variables in ``python/setup.py`` as necessary. 
#. Run ``python setup.sty install`` in the ``python`` directory to compile the package and install it on your system.

#############
Configuration
#############

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



###############
Troubleshooting
###############

#. If it becomes necessary to change the location where `setuptools` 
   places the compiled module, add a file called ``setup.cfg`` 
   with content 

   .. code-block:: none
   
    [install]
    prefix=<target directory where you want the module to be placed>

   in the ``python`` directory. Make sure that the new target directory is in Python's load path.

#. PyTorch requires C++ extensions to be compiled against the same version of CUDA that it  
   itself was compiled with. If this becomes an issue, it might be necessary to install an 
   alternative version of CUDA on your system and force `setuptools` to use that version by setting 
   the ``CUDA_HOME`` enironment variable, as, e.g. 

   .. code-block:: none
   
    export CUDA_HOME=/usr/local/cuda-11.3


############
Known issues
############

GPU functionality is not fully tested.


 
##### 
Usage 
#####

To load the ``gelib`` module in Python, use the command ``import gelib``.

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


