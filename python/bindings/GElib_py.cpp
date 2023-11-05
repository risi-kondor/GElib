
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

#define _WITH_FAKE_GRAD

#include "GElib_base.cpp"
#include "GElibSession.hpp"
#include "diff_class.hpp"

#include "RtensorObj.hpp"

#include "WignerMatrix.hpp"
#include "SO3type.hpp"
//#include "SO3bitype.hpp"
//#include "SO3part.hpp"
//#include "SO3vec.hpp"
//#include "SO3partArray.hpp"
//#include "SO3vecArray.hpp"

#include "SO3partB.hpp"
#include "SO3vecB.hpp"
#include "SO3mvec.hpp"
#include "SO3weights.hpp"
#include "SO3mweights.hpp"
#include "SO3partB_array.hpp"
#include "SO3vecB_array.hpp"
#include "SO3CGtensor.hpp"

//#include "SO3partC.hpp"
//#include "SO3vecC.hpp"
//#include "SO3partArrayC.hpp"
//#include "SO3vecArrayC.hpp"

//#include "SO3bipart.hpp"
//#include "SO3bivec.hpp"
//#include "SO3bipartArray.hpp"
//#include "SO3bivecArray.hpp"

#include "SO3partC.hpp"

#include "CtensorConvolve2d.hpp"
//#include "CtensorConvolve2dSparse.hpp"
#include "CtensorConvolve3d.hpp"
#include "CtensorConvolve3d_back0.hpp"

//std::default_random_engine rndGen;

GElib::GElibSession session;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


  using namespace cnine;
  using namespace GElib;
  namespace py=pybind11;

  py::options options;
  //options.disable_function_signatures();

  m.def("version",[](){cout<<_GELIB_VERSION<<endl;});

  py::class_<SO3element>(m,"SO3element")

    .def(pybind11::init<>(),"")
    .def(pybind11::init<double,double,double>(),"")

    .def_static("identity",&SO3element::identity)
    .def_static("uniform",&SO3element::uniform)

    .def("rho",[](const SO3element& g, const int l){
	return cnine::CtensorObj(WignerMatrix<float>(l,g));
      })

    .def("str",&SO3element::str,"Print the SO3element to string.")
    .def("__str__",&SO3element::str,"Print the SO3element to string.")
    .def("__repr__",&SO3element::str,"Print the SO3element to string.");




  //m.def("CGtransform",static_cast<SO3type (*)(const SO3bitype&, const int)>(&CGtransform),
  //py::arg("x"),py::arg("maxl")=-1);


  m.def("add_WignerMatrix_to",
    static_cast<void(*)(cnine::CtensorB&, const int, const double, const double, const double)>(&add_WignerMatrix_to));

  m.def("add_CGmatrix_to",[](cnine::RtensorObj& R, const int l1, const int l2, const int l){
      R+=SO3CGmatrix(l1,l2,l);
    });

  m.def("add_CGtensor_to",[](cnine::RtensorObj& R, const int l1, const int l2, const int l){
      R+=SO3CGtensor(l1,l2,l);
    });

  #include "SO3type_py.cpp"
  #include "SO3partB_py.cpp"
  #include "SO3vecB_py.cpp"
  #include "SO3mvec_py.cpp"
  #include "SO3weights_py.cpp"
  #include "SO3mweights_py.cpp"

  #include "SO3partB_array_py.cpp"
  #include "SO3vecB_array_py.cpp"

  //#include "SO3part_py.cpp"
  //#include "SO3vec_py.cpp"
  //#include "SO3partArray_py.cpp"
  //#include "SO3vecArray_py.cpp"

  //#include "SO3bipart_py.cpp"
  //#include "SO3bivec_py.cpp"
  //#include "SO3bipartArray_py.cpp"
  //#include "SO3bivecArray_py.cpp"

  #include "SO3partC_py.cpp"

}

