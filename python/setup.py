import sys,os
import torch 
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import time 

#os.environ['CUDA_HOME']='/usr/local/cuda'
#os.environ["CC"] = "clang"


# --- User settings ------------------------------------------------------------------------------------------

compile_with_cuda=True

copy_warnings=True
torch_convert_warnings=True 


# ------------------------------------------------------------------------------------------------------------

cwd = os.getcwd()

_include_dirs=[cwd+'/../../cnine/include',
               cwd+'/../../cnine/include/cmaps',
               cwd+'/../../cnine/objects/scalar',
               cwd+'/../../cnine/objects/tensor',
               cwd+'/../../cnine/objects/tensor_views',
               cwd+'/../../cnine/objects/tensor_array',
               cwd+'/../../cnine/objects/tensor_array/cell_ops',
               cwd+'/../include',
               cwd+'/../combinatorial',
               cwd+'/../objects/SO3',
               cwd+'/../objects/SO3/cell_ops',
               cwd+'/../objects/SO3/functions'
               ]

_cxx_compile_args=['-std=c++14',
                  '-Wno-sign-compare',
                  '-Wno-deprecated-declarations',
                  '-Wno-unused-variable',
                  #'-Wno-unused-but-set-variable',
                  '-Wno-reorder',
                  '-Wno-reorder-ctor',
                  '-Wno-overloaded-virtual',
                  '-D_WITH_ATEN',
                  '-DCNINE_RANGE_CHECKING',
                  '-DCNINE_SIZE_CHECKING',
                  '-DCNINE_DEVICE_CHECKING'
                  ]

_nvcc_compile_args=['-D_WITH_CUDA',
                   '-D_WITH_CUBLAS',
                   '-D_DEF_CGCMEM',
                   #'-rdc=true'
                   ]


if copy_warnings:
    _cxx_compile_args.extend([
        '-DCNINE_COPY_WARNINGS',
        '-DCNINE_ASSIGN_WARNINGS',
        '-DCNINE_MOVE_WARNINGS',
        '-DCNINE_MOVEASSIGN_WARNINGS'
        ])

if torch_convert_warnings:
    _cxx_compile_args.extend([
        '-DCNINE_ATEN_CONVERT_WARNINGS'
        ])
    
if compile_with_cuda:
    _cxx_compile_args.extend(['-D_WITH_CUDA','-D_WITH_CUBLAS'])
    
_depends=['setup.py',
          'GElib_py.cpp',
          'SO3part_py.cpp',
          'SO3vec_py.cpp',
          'SO3partArray_py.cpp',
          'SO3vecArray_py.cpp',
          'build/*/*'
          ] 


# ---- Compilation commands ----------------------------------------------------------------------------------


if compile_with_cuda:
    setup(name='gelib_base',
          ext_modules=[CUDAExtension('gelib_base', [ 
          '../../cnine/include/Cnine_base.cu',
          '../../cnine/cuda/TensorView_accumulators.cu',
          '../cuda/SO3CGproducts_combo.cu',
          #'../cuda/GElib_base.cu',
          #'../cuda/SO3partA_CGproduct.cu',
          #'../cuda/SO3partA_DiagCGproduct.cu',
          #'../cuda/SO3partB_addCGproduct.cu',
          #'../cuda/SO3partB_addCGproduct_back0.cu',
          #'../cuda/SO3partB_addCGproduct_back1.cu',
          #'../cuda/SO3Fpart_addFproduct.cu',
          #'../cuda/SO3Fpart_addFproduct_back0.cu',
          #'../cuda/SO3Fpart_addFproduct_back1.cu',
          'GElib_py.cpp'
          ],
                                     include_dirs=_include_dirs,
                                     extra_compile_args={
                                         'nvcc': _nvcc_compile_args,
                                         'cxx': _cxx_compile_args},
                                     depends=_depends
                                     )],
          cmdclass={'build_ext': BuildExtension})
else:
    setup(name='gelib_base',
          ext_modules=[CppExtension('gelib_base', ['GElib_py.cpp'],
                                    include_dirs=_include_dirs,
                                    extra_compile_args = {
                                        'cxx': _cxx_compile_args},
                                    depends=_depends
                                    )],
          cmdclass={'build_ext': BuildExtension})

print("Compilation finished:",time.ctime(time.time()))


# ------------------------------------------------------------------------------------------------------------
