# GElib

GElib is a C++/CUDA library for building equivariant neural networks. The library can be used as a pure 
C++ library or as a C++ extension for PyTorch. 
GElib uses [`cnine`](https://github.com/risi-kondor/cnine) as its backend. 
Documentation for the C++ API can be found in the 'doc' directory. 
Documentation for the PyTorch API is at https://risi-kondor.github.io/GElib/.

GElib is released under a combination of the Mozilla Public License v. 2.0. and a custom noncommercial license to be found in the file NONCOMMERCIAL.TXT. 
The latter also applies to the `cnine` component. Commercial use requires a licence from the 
author(s). 
<br><br>

# Default installation as a PyTorch C++ extension

GElib is distributed in source code format, therefore to install it as a C++ extension you must have a working C++ 
compiler supporting C++17 (or higher) on your system (GCC or clang recommended). You also need PyTorch. 
If these requirements are satified, in most cases GElib can be installed simply by running  
```bash
   pip install gelib
   ``` 
or 
```bash
   pip3 install gelib
   ``` 

## Installation with CUDA support 

Compiling GElib with CUDA enabled is more complicated because it also requires the CUDA development environment to be present 
on your system. Further, you must use the same CUDA version as the version that PyTorch was compiled against. If, for example, 
your PyTorch needs CUDA 12.6, which is installed at `/usr/local/cuda-12.6`, you need to point the installation script to it with 
```bash
   export CUDA_HOME="/usr/local/cuda-12.6"
   ```
To compile with CUDA enabled you must also set  
```bash
   export WITH_CUDA="TRUE"
   ```
before running `pip install gelib` command. 
<br><br>

# Fallback installation method as a C++ extension

The fallback installation method is to clone the library with 
```bash
   git clone --recurse-submodules git@github.com:risi-kondor/GElib.git
   ```
and install it manually by running 
```bash
   pip install -e . --no-build-isolation
   ```
in its root directory. 
The `--recurse-submodules` flag ensures that a copy of `cnine` is bundled inside the GElib directory structure. 
The `--no-build-isolation` flag forces GElib to be built using versions of PyTorch etc. already present in your environment 
rather than temporary versions downloaded during the build process. This can help avoid version conflicts. 
The CUDA component of the library is enabled the same way as in the default installation method.
<br><br>

## Credits 

Lead developer: Risi Kondor

Contributors: Erik Henning Thiede, Ryan Keane

