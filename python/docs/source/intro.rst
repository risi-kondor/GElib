GElib is a C++ library with a CUDA backend for computations related to 
the rotation group :math:`\mathrm{SO}(3)`, specifically for building SO(3)-equivariant 
neural networks. 
GElib is developed by Risi Kondor at the University of Chicago. 
GElib is released under the 
`Mozilla public license v.2.0 <https://www.mozilla.org/en-US/MPL/2.0/>`_.   

This document provides documentation for GElib's Python interface. Not all features in the C++ library 
are available through this interface. The documentation of the C++ API can be found in pdf format 
in the package's ``doc`` directory.

########
Features
########

#. Classes to store and manipulate SO(3)-equivariant vectors
#. Fast implementation of Clebsch-Gordan transforms on both the CPU and the GPU
#. Facilities for operating on arrays of SO(3)-equivariant vectors in parallel, even in irregular patterns (graphs)
#. Full support for automatic differentiation
#. Interoperability with PyTorch

 
############
Installation
############

Installing GElib requires the following:

#. C++14 or higher
#. Python
#. PyTorch
#. cnine (see below) 

To install GElib follow these steps:

#. Download the `cnine <https://github.com/risi-kondor/cnine>`_  and 
   `GElib <https://github.com/risi-kondor/GElib>`_ libraries. 
#. Edit the file ``config.txt`` as necessary, in particular, make sure that ``CNINE_ROOT`` points to the root 
   of the *cnine* package on your system. 
#. Run ``python setup.sty install`` in the ``python`` directory to compile the package and install it on your 
   system.

##### 
Usage 
#####

GElib has two distinct interfaces implemented in two different modules:

#. To use the library *without* PyTorch's autodiff functionality, load the library with ``import gelib_base as gelib``. 
#. To use the library *with* automatic differentiation, load the library with ``import gelib_torch as gelib``. 

The two modules use identical syntax, therefore the following description of their usage applies to both. 
The backend implementation of the two modules however is quite different: whereas ``gelib_base`` is mostly  
just a wrapper for the underlying C++ classes, 
``gelib_torch`` 's core classes are derived from ``torch.tensor`` for interoperability with ``torch.autodiff``. 
Inevitably, the latter approach incurs some performanvce overhead.  
``gelib_torch`` is built on ``gelib_base``, so the two modules can also be used together.   

############
Known issues
############

GPU functionality is temporarily disabled. 