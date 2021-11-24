GElib is a C++ library with a CUDA backend for performing certain operations related to 
the rotation group :math:`\mathrm{SO}(3)`. 
GElib is developed by Risi Kondor at the University of Chicago. 
GElib is released under the 
`Mozilla public license v.2.0 <https://www.mozilla.org/en-US/MPL/2.0/>`_.   

This document provides documentation for GElib's Python interface. Not all features in the C++ library 
are available through this interface. The documentation of the C++ API can be found in pdf format 
in the package's ``doc`` directory.

########
Features
########


 
############
Installation
############

Installing GElib requires the following:

#. C++11 or higher
#. Python
#. pybind11 (comes with PyTorch)
#. cnine (see below) 
#. PyTorch (optional)

To install GElib follow these steps:

#. Download the `cnine <https://github.com/risi-kondor/cnine>`_  and 
   `GElib <https://github.com/risi-kondor/GElib>`_ libraries. 
#. Edit the file ``config.txt`` as necessary, in particular, make sure that ``CNINE_ROOT`` points to the root 
   of the *cnine* package on your system. 
#. Run ``python setup.sty install`` in the ``python`` directory to compile the package and install it on your 
   system.
 
To use GElib from Python, load the corresponding module the usual way with ``import GElib``. 
In the following we assume that ``from GElib import *`` has been called,  
obviating the need to prefix all GElib classes with ``GElib.``. 
For matrix/tensor functionality the ``cnine`` module must also be loaded. 

############
Known issues
############

GPU functionality is temporarily disabled. 