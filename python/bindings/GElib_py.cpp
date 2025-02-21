
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include <torch/torch.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

//#define _WITH_FAKE_GRAD

#include "GElib_base.cpp"
#include "GElibSession.hpp"
//#include "diff_class.hpp"

//#include "WignerMatrix.hpp"
#include "SO3type.hpp"
#include "SO3part.hpp"
#include "SO3vec.hpp"
#include "SO3functions.hpp"
//#include "SO3mvec.hpp"
//#include "SO3weights.hpp"
//#include "SO3mweights.hpp"
//#include "SO3partB_array.hpp"
//#include "SO3vecB_array.hpp"
//#include "SO3CGtensor.hpp"

GElib::GElibSession session;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


  using namespace cnine;
  using namespace GElib;
  namespace py=pybind11;

  py::options options;

  m.def("version",[](){cout<<_GELIB_VERSION<<endl;});

  typedef cnine::TensorView<float> tensorf;
  typedef cnine::TensorView<complex<float> > tensorc;

<<<<<<< HEAD
  //#include "SO3part_py.cpp"
  //#include "SO3vec_py.cpp"
  //#include "SO3partArray_py.cpp"
  //#include "SO3vecArray_py.cpp"

  //#include "SO3bipart_py.cpp"
  //#include "SO3bivec_py.cpp"
  //#include "SO3bipartArray_py.cpp"
  //#include "SO3bivecArray_py.cpp"
=======
  #include "SO3type_py.cpp"
  #include "SO3part_py.cpp"
  #include "SO3vec_py.cpp"
  //#include "SO3mvec_py.cpp"
  //#include "SO3weights_py.cpp"
  //#include "SO3mweights_py.cpp"
>>>>>>> dev2

  //#include "SO3partB_array_py.cpp"
  //#include "SO3vecB_array_py.cpp"


}

