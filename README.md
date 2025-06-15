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

## Installing GElib as a Python module

GElib is distributed in source code format, therefore to install it you must have a working C++ 
compiler supporting C++17 (or higher) on your system. You also need PyTorch. 
If these requirements are satified, in most cases GElib can be installed simply by running  
```bash
   pip install gelib
   ``` 
or 
```bash
   pip3 install gelib
   ``` 

To install the package with CUDA support, set the `WITH_CUDA` environment variable to `TRUE`:
```bash
   export WITH_CUDA="TRUE"
   ```
*before* running the above command. 
Installing with CUDA requires the CUDA development environment to be present 
on your system and its version must match that which PyTorch was compiled against. 
<br><br>

### Fallback installation method

The fallback installation method is to clone the library with 
```bash
   git clone --recurse-submodules git@github.com:risi-kondor/GElib.git
   ```
and install it manually by running `pip install -e .` in its root directory. 
The `--recurse-submodules` flag ensures that a copy of `cnine` is bundled inside the GElib directory structure. 
<br><br>

## Credits 

Lead developer: Risi Kondor

Contributors: Erik Henning Thiede, Ryan Keane

