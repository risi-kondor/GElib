
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


// Define TORCH_EXTENSION_NAME before including torch/extension.h
// This is crucial for PyTorch C++ extensions.
#define TORCH_EXTENSION_NAME gelib_base
#include <torch/extension.h> // For PyTorch C++ extensions and pybind11 type casters
#include <pybind11/stl.h>
#include <pybind11/complex.h>

//#define _WITH_FAKE_GRAD

#include "GElib_base.hpp" // Use the header file
#include "GElibSession.hpp"
//#include "diff_class.hpp"

#include "GatherMapB.hpp" // temporary 

#include "SO3element.hpp"
#include "SO3irrep.hpp"
#include "SO3type.hpp"
#include "SO3part.hpp"
#include "SO3vec.hpp"
#include "SO3functions.hpp"


GElib::GElibSession session;


PYBIND11_MODULE(gelib_base, m) {


  using namespace cnine;
  using namespace GElib;
  namespace py=pybind11;

  py::options options;

  m.def("version",[](){cout<<_GELIB_VERSION<<endl;});

  typedef cnine::TensorView<float> tensorf;
  typedef cnine::TensorView<complex<float> > tensorc;

  #include "gather_map_py.cpp"

  #include "SO3element_py.cpp"
  #include "SO3irrep_py.cpp"
  #include "SO3type_py.cpp"
  #include "SO3part_py.cpp"
  #include "SO3vec_py.cpp"

}

