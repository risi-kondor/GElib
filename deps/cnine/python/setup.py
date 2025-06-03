import sys,os
import torch
from setuptools import setup
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
import time
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

    compile_with_cuda = interpret_bool_string(os.environ.get("WITH_CUDA", False))
    copy_warnings= interpret_bool_string(os.environ.get("COPY_WARNING", False))
    torch_convert_warnings=interpret_bool_string(os.environ.get("TORCH_CONVERT_WARNINGS", False))

    if compile_with_cuda: 
        print("CUDA found at "+os.environ['CUDA_HOME'])
    else:
        print("No CUDA found, installing without GPU support.")

    cwd = os.getcwd()

    _include_dirs=[
        cwd+'/../include',
        cwd+'/../algorithms',
        cwd+'/../combinatorial',
        cwd+'/../containers',
        cwd+'/../hpc',
        cwd+'/../math',
        cwd+'/../matrices',
        cwd+'/../tensors',
        cwd+'/../utility',
        cwd+'/../wrappers',
        #cwd+'/../include/cmaps',
        #cwd+'/../legacy/scalar',
        #cwd+'/../legacy/matrix',
        #cwd+'/../legacy/tensor',
        #cwd+'/../legacy/backendA',
        #cwd+'/../legacy/backendB',
        cwd+'/../tensor_views',
        #cwd+'/../legacy/tensor_array',
        #cwd+'/../legacy/tensor_array/cell_maps',
        #cwd+'/../legacy/tensor_array/cell_ops',
        #cwd+'/../legacy/labeled',
        #cwd+'/../legacy/ntensor',
        #cwd+'/../legacy/ntensor/functions'
        ]

    _nvcc_compile_args=[
        '-D_WITH_CUDA',
        '-D_WITH_CUBLAS',
        '-DWITH_FAKE_GRAD'
        ]

    _cxx_compile_args=['-std=c++17',
                       '-D_WITH_ATEN',
                       '-DCNINE_RANGE_CHECKING',
                       '-DCNINE_SIZE_CHECKING',
                       '-DCNINE_DEVICE_CHECKING',
                       '-DWITH_FAKE_GRAD'
                      ]
    # Adding compiler spcific flags
    if os.name == "posix":
        _cxx_compile_args += ['-Wno-sign-compare',
                              '-Wno-deprecated-declarations',
                              '-Wno-unused-variable',
                              '-Wno-unused-but-set-variable',
                              '-Wno-reorder',
                              '-Wno-reorder-ctor',
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
        _cxx_compile_args.extend([
            '-D_WITH_CUDA',
            '-D_WITH_CUBLAS'
            ])

    _depends=['setup.py',
              'bindings/*.cpp',
              #'cnine_py.cpp',
              #'rtensor_py.cpp',
              #'ctensor_py.cpp',
              #'rtensorarr_py.cpp',
              #'ctensorarr_py.cpp',
              #'cmaps_py.cpp',
              'build/*/*'
              ]


    # ---- Compilation commands ----------------------------------------------------------------------------------


    if compile_with_cuda:
        ext_modules=[CUDAExtension('cnine_base',
                                   ['bindings/cnine_py.cpp',
                                    '../include/Cnine_base.cu',
                                    '../cuda/RtensorUtils.cu',
                                    '../cuda/RtensorReduce.cu',
                                    '../cuda/RtensorEinsumProducts.cu'],
                                   include_dirs=_include_dirs,
                                   extra_compile_args = {
                                       'nvcc': _nvcc_compile_args,
                                       'cxx': _cxx_compile_args},
                                   depends=_depends,
                                   )]
    else:
        ext_modules=[CppExtension('cnine_base',
                                  ['bindings/cnine_py.cpp'],
                                  include_dirs=_include_dirs,
                                  extra_compile_args = {
                                      'cxx': _cxx_compile_args},
                                  depends=_depends,
                                  )]


    setup(name='cnine',
          ext_modules=ext_modules,
          packages=find_packages('src'),
          package_dir={'': 'src'},
          py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
          include_package_data=True,
          zip_safe=False,
        cmdclass={'build_ext': BuildExtension}
    )


# ------------------------------------------------------------------------------------------------------------
    

if __name__ == "__main__":
    main()

print("Compilation finished:",time.ctime(time.time()))


#os.environ['CUDA_HOME']='/usr/local/cuda' #doesn't work, need explicit export
#os.environ["CC"] = "clang"
#CUDA_HOME='/usr/local/cuda'
#print(torch.cuda.is_available())
