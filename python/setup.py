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
                                              '../../cnine/include/cmaps',
                                              '../../cnine/objects/scalar',
                                              '../../cnine/objects/tensor',
                                              '../../cnine/objects/tensor_array',
                                              '../../cnine/objects/tensor_array/cell_ops',
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
                                                               '-D_WITH_ATEN',
                                                                    '-DCNINE_COPY_WARNINGS',
                                                               '-DCNINE_ASSIGN_WARNINGS',
                                                                                '-DCNINE_MOVE_WARNINGS',
                                                               '-DCNINE_MOVEASSIGN_WARNINGS',
                                                               '-DCNINE_RANGE_CHECKING'
                                                               ]},
                                 depends=['setup.py',
                                          'GElib_py.cpp',
                                          'SO3part_py.cpp',
                                          'SO3vec_py.cpp',
                                          'SO3partArray_py.cpp',
                                          'SO3vecArray_py.cpp',
                                          'build/*/*'
                                          ])], 
      cmdclass={'build_ext': BuildExtension}
      )

