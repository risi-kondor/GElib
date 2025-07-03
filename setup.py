import os
import torch
from setuptools import setup
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from torch.utils.cpp_extension import include_paths, library_paths
import time
from os.path import splitext
from os.path import basename
from glob import glob
from typing import Union, Tuple


def interpret_bool_string(string:Union[str,bool], _true_values:Tuple[str] = ("TRUE", "ON"), _false_values:Tuple[str] = ("FALSE", "OFF")):
    if isinstance(string, bool):
        return string
    if string.strip().upper() in _true_values:
        return True
    if string.strip().upper() in _false_values:
        return False
    raise ValueError(f"String {string} cannot be interpreted as True or False. Any upper/lower-case version of {_true_values} is True, {_false_values} is False. {string} was neither.")


def main():

    print("-------------------------------------------------------------")
    compile_with_cuda = interpret_bool_string(os.environ.get("WITH_CUDA", False))
    if compile_with_cuda:
        print("Compiling GElib with CUDA support")
    else: 
        print("Compiling GElib without CUDA support")

    copy_warnings = False
    torch_convert_warnings = True

    cwd = os.getcwd()+"/"
    cnine_folder = "deps/cnine/"
    ext_cuda_folder = "cuda/"

    _include_dirs = [cwd + cnine_folder + '/include',
		     cwd + cnine_folder + '/combinatorial',
		     cwd + cnine_folder + '/containers',
		     cwd + cnine_folder + '/hpc',
		     cwd + cnine_folder + '/math',
		     cwd + cnine_folder + '/matrices',
		     cwd + cnine_folder + '/utility',
		     cwd + cnine_folder + '/tensors',
		     cwd + cnine_folder + '/tensors/functions',
		     cwd + cnine_folder + '/wrappers',
                     cwd + cnine_folder + '/include/cmaps',
                     cwd + cnine_folder + '/objects/matrix',
                     cwd + cnine_folder + '/tensor_views',
                     cwd + 'include',
                     cwd + 'cuda',
                     cwd + 'core',
                     cwd + 'SO3',
                     cwd + 'O3'
                     ]


    _cxx_compile_args = ['-std=c++17',
                         '-Wno-sign-compare',
                         '-Wno-deprecated-declarations',
                         '-Wno-unused-variable',
                         '-Wno-reorder',
                         '-Wno-reorder-ctor',
                         '-Wno-overloaded-virtual',
                         '-D_WITH_ATEN',
                         '-DCNINE_RANGE_CHECKING',
                         '-DCNINE_SIZE_CHECKING',
                         '-DCNINE_DEVICE_CHECKING',
                         '-DGELIB_RANGE_CHECKING',
                         '-DWITH_FAKE_GRAD'
                         ]

    _nvcc_compile_args = ['-D_WITH_CUDA',
                          '-D_WITH_CUBLAS',
                          '-D_DEF_CGCMEM',
                          '-DGELIB_RANGE_CHECKING',
                          '-DWITH_FAKE_GRAD',
                          '-std=c++17',
                          '-rdc=true'
                          ]

    if copy_warnings:
        _cxx_compile_args.extend([
            '-DCNINE_COPY_WARNINGS',
            '-DCNINE_ASSIGN_WARNINGS',
            '-DCNINE_MOVE_WARNINGS',
            '-DCNINE_MOVEASSIGN_WARNINGS',
            '-DGELIB_COPY_WARNINGS',
            '-DGELIB_MOVE_WARNINGS',
            '-DGELIB_CONVERT_WARNINGS'
        ])

    if torch_convert_warnings:
        _cxx_compile_args.extend([
            '-DCNINE_ATEN_CONVERT_WARNINGS'
        ])

    if compile_with_cuda:
        _cxx_compile_args.extend(['-D_WITH_CUDA', '-D_WITH_CUBLAS'])

    _depends = ['setup.py',
                'src/gelib.cpp',
                'bindings/*.cpp'
                ]


    # ---- Compilation commands ----------------------------------------------------------------------------------


    if compile_with_cuda:
        ext_modules = [CUDAExtension('gelib_base', [
            cnine_folder+'include/Cnine_base.cu',
            cnine_folder+'cuda/TensorView_assign.cu',
            'cuda/GElib_base.cu',
            'cuda/SO3part_addCGproduct.cu',
            'cuda/SO3part_addCGproduct_back0.cu',
            'cuda/SO3part_addCGproduct_back1.cu',
            'python/bindings/GElib_py.cpp'
            ],
            include_dirs=_include_dirs,
            extra_compile_args={
            'nvcc': _nvcc_compile_args,
            'cxx': _cxx_compile_args},
            depends=_depends
        )]
    else:
        ext_modules=[CppExtension('gelib_base',
            ['python/bindings/GElib_py.cpp'],
            include_dirs=_include_dirs,
            #library_dirs=library_paths(),
            #libraries=['c10'],
            extra_compile_args={'cxx': _cxx_compile_args},
            depends=_depends
        )]

    setup(name='gelib',
          ext_modules=ext_modules,
          packages=find_packages('python/src'),
          package_dir={'': 'python/src'},
          py_modules=[splitext(basename(path))[0] for path in glob('python/src/*.py')],
          include_package_data=True,
          zip_safe=False,
          cmdclass={'build_ext': BuildExtension})

    print("Compilation finished:", time.ctime(time.time()))


    # ------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
