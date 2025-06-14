# GElib

GElib is a C++/CUDA library for building equivariant neural networks. The library can be used as a pure 
C++ library or as a C++ extension for PyTorch. 
GElib uses [`cnine`](https://github.com/risi-kondor/cnine) as its backend. 
Documentation for the C++ API can be found in the 'doc' directory. 
Documentation for the PyTorch API is at https://risi-kondor.github.io/GElib/.

GElib is released under a combination of the Mozilla Public License v. 2.0. and a custom noncommercial license to be found in the file NONCOMMERCIAL.TXT. 
The latter also applies to the `cnine` component. Commercial use requires a licence from the 
author(s). 


## Installing GElib as a Python module

GElib is distributed in source code format, therefore to install it you need to have a working C++ 
compiler to be present on your system. The compiler must support the C++17 standard (or higher). 
You also need PyTorch. 

To install GElib with CUDA support, you need to have the CUDA development environment to be installed on your system. 

1. Clone the package with  
```bash
   git clone --recurse-submodules git@github.com:risi-kondor/GElib.git
   ```
(The `--recurse-submodules` flag ensures that `cnine` is fetched automatically.)

2. Move to the `python` subdirectory:
```bash
   cd GElib/python
   ```
3. If you wish to install GElib with CUDA support set the `WITH_CUDA` environment variable to `TRUE`:
```bash
   export WITH_CUDA="TRUE"
   ```
4. Compile and install the package with 
```bash
   pip install -e .
   ``` 
   or 
```bash
   pip3 install -e .
   ``` 

## Credits 

Lead developer: Risi Kondor

Contributors: Erik Henning Thiede, Ryan Keane

