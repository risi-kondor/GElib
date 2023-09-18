import os
import torch
from setuptools import setup
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import time
from os.path import splitext
from os.path import basename
from glob import glob


def main():

    # --- User settings ------------------------------------------------------------------------------------------
    # os.environ['CUDA_HOME']='/usr/local/cuda'
    #os.environ["CC"] = "clang"

    compile_with_cuda = False 
    # compile_with_cuda = False

    copy_warnings = False
    torch_convert_warnings = True

    # ------------------------------------------------------------------------------------------------------------
    
#    if 'CUDA_HOME' in os.environ:
#        print("CUDA found at "+os.environ['CUDA_HOME'])
#    else:
#        print("No CUDA found, installing without GPU support.")
#        compile_with_cuda=False

    cwd = os.getcwd()
    root_dir="/../../.."
    #cnine_dir = cwd+"/../../../../cnine"
    cnine_dir = "../../../../cnine"
    snob_dir = "/../../../../Snob2"

    _include_dirs = [#'/usr/local/include',
		     
                     cnine_dir,
                     cnine_dir + '/include',
                     cnine_dir + '/modules',
		     cnine_dir + '/combinatorial',
		     #cnine_dir + '/containers',
		     cnine_dir + '/math',
		     cnine_dir + '/wrappers',
                     #cnine_dir + '/include/cmaps',
                     cnine_dir + '/objects/scalar',
                     #cnine_dir + '/objects/matrix',
                     cnine_dir + '/objects/tensor',
                     cnine_dir + '/objects/backendA',
                     cnine_dir + '/objects/backendB',
                     cnine_dir + '/objects/tensor_views',
                     cnine_dir + '/objects/tensor_views/functions',
                     #cnine_dir + '/objects/tensor_array',
                     #cnine_dir + '/objects/tensor_array/cell_maps',
                     #cnine_dir + '/objects/tensor_array/cell_ops',
                     #cnine_dir + '/objects/labeled',
                     #cnine_dir + '/objects/ntensor',
                     #cnine_dir + '/objects/ntensor/functions',

                     cwd + snob_dir+'/include',
                     cwd + snob_dir+'/combinatorial',
                     cwd + snob_dir+'/Sn',

                     cwd + root_dir+'/include',
                     cwd + root_dir+'/cuda',
                     cwd + root_dir+'/objects/SO2',
                     cwd + root_dir+'/objects/SO2/functions',
                     cwd + root_dir+'/objects/SO3',
                     cwd + root_dir+'/objects/SO3/cell_ops',
                     cwd + root_dir+'/objects/SO3/functions',
                     cwd + root_dir+'/objects/SO3n',
                     cwd + root_dir+'/objects/SO3n/functions',
                     cwd + root_dir+'/objects/prod'
                     ]


    _cxx_compile_args = ['-std=c++17',
                         '-Wno-sign-compare',
                         '-Wno-deprecated-declarations',
                         '-Wno-unused-variable',
                         # '-Wno-unused-but-set-variable',
                         '-Wno-reorder',
                         '-Wno-reorder-ctor',
                         '-Wno-overloaded-virtual',
                         '-D_WITH_ATEN',
                         '-DCNINE_RANGE_CHECKING',
                         '-DCNINE_SIZE_CHECKING',
                         '-DCNINE_DEVICE_CHECKING',
                         '-DGELIB_RANGE_CHECKING',
                         '-DWITH_EIGEN'
                         ]

    _nvcc_compile_args = ['-D_WITH_CUDA',
                          '-D_WITH_CUBLAS',
                          '-D_DEF_CGCMEM',
                          '-DGELIB_RANGE_CHECKING',
                          '-DWITH_FAKE_GRAD'
                          # '-rdc=true'
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
                'bindings/*.cpp'
                ]


    # ---- Compilation commands ----------------------------------------------------------------------------------


    if compile_with_cuda:
        ext_modules = [CUDAExtension('geprod_base', [
            'bindings/GEprod_py.cpp'
        ],
            include_dirs=_include_dirs,
            extra_compile_args={
            'nvcc': _nvcc_compile_args,
            'cxx': _cxx_compile_args},
            depends=_depends
        )]
    else:
        ext_modules = [CppExtension('geprod_base', ['bindings/GEprod_py.cpp'],
                                    include_dirs=_include_dirs,
                                    # sources=sources,
                                    extra_compile_args={
            'cxx': _cxx_compile_args},
            depends=_depends
        )]

    setup(name='geprod',
          ext_modules=ext_modules,
          packages=find_packages('src'),
          package_dir={'': 'src'},
          py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
          include_package_data=True,
          zip_safe=False,
          cmdclass={'build_ext': BuildExtension})

    # print("Compilation finished:", time.ctime(time.time()))

    # ------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
