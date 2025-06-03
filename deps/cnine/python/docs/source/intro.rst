############
Introduction
############

********
Features
********

#. Support for real and complex valued tensors.
#. Transparent block level parallelization on GPUs using the tensor array data structures. 
#. Ability to execute custom CUDA kernels in parallel across many cells of a tensor array, even in irregular patterns. 

************
Installation
************

.. 
 `cnine` can be used with or without PyTorch. However, the installation script uses PyTorch's 
 cpp-extension facility. 
 Therefore, installation requires the following:

Installing `cnine` requires the following: 

#. C++11 or higher
#. PyTorch
#. CUDA and CUBLAS for GPU functionality 

`cnine` is easiest to install with ``pip``:

#. Download `cnine` from `github <https://github.com/risi-kondor/cnine>`_. 
#. Edit the user configurable variables in ``python/setup.py`` as necessary. 
#. Run the command ``pip install -e .`` in the ``python`` directory to compile the package and install it on your 
   system.
 
To use `cnine` from Python, load the module the usual way with ``import cnine``. 
If `cnine` was compiled with GPU support, you must first ``import torch``. 

.. 
 In the following we assume that the command ``from cnine import *`` has been issued,  
 obviating the need to prefix all `cnine` classes and funnctions with ``cnine.``.

*************
Configuration
*************

The `cnine` installation can be configured by setting the corresponding variables in ``python/setup.py``.

``compile_with_cuda``
  If set to ``True``, `cnine` will be compiled with GPU suport. This requires a working CUDA and CUBLAS installation 
  on your system and PyTorch itself having been compiled with CUDA enabled. To make sure that the appropriate 
  runtime libraries are loaded, you must always import ``torch`` before importing ``cnine``.

``copy_warnings``
  If set to ``True``, `cnine` will print a message to the terminal whenever a tensor or tensor array object 
  is copied or move-copied. This option is useful for code optimization. 

``torch_convert_warnings`` 
  If set to ``True``, `cnine` will print a message to the terminal whenver a tensor is explicitly converted 
  (as opposed to just forming a tensor view) to/from PyTorch format. 


************
Known issues
************

GPU functionality is currently not fully tested.  

***************
Troubleshooting
***************

.. 
   If it becomes necessary to change the location where `setuptools` 
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

