import sys,os
import torch 
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

#os.environ['CUDA_HOME']='/usr/local/cuda'
os.environ["CC"] = "clang"
cwd = os.getcwd()

#CUDA_HOME='/usr/local/cuda'
#print(torch.cuda.is_available())

setup(name='GElib',
      ext_modules=[CppExtension('GElib', ['GElib_py.cpp'],
                                include_dirs=['../../cnine/include',
                                              '../../cnine/objects/scalar',
                                              '../../cnine/objects/tensor',
                                              '../../cnine/objects/tensor_array',
                                              cwd+'/../include',
                                              cwd+'/../combinatorial',
                                              cwd+'/../objects/SO3',
                                              cwd+'/../objects/SO3/cell_ops'],
                                 extra_compile_args = {
                                                       'cxx': ['-std=c++14',
                                                               '-Wno-sign-compare',
                                                               '-Wno-deprecated-declarations',
                                                               '-Wno-unused-variable',
                                                               '-Wno-reorder-ctor',
                                                               '-Wno-reorder',
                                                               ]},
                                 depends=['setup.py',
                                          'GElib_py.cpp'
                                          ])], 
      cmdclass={'build_ext': BuildExtension}
      )

