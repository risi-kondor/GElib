
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

#include "GElib_base.cpp"
#include "GElibSession.hpp"

#include "RtensorObj.hpp"

#include "WignerMatrix.hpp"
#include "SO3type.hpp"
#include "SO3part.hpp"
#include "SO3vec.hpp"
//#include "SO3partArray.hpp"
//#include "SO3vecArray.hpp"

#include "SO3partB.hpp"
#include "SO3vecB.hpp"
#include "SO3Fvec.hpp"
#include "SO3partD.hpp"
#include "SO3vecD.hpp"


//std::default_random_engine rndGen;

GElib::GElibSession session;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


  using namespace cnine;
  using namespace GElib;
  namespace py=pybind11;

  py::options options;
  //options.disable_function_signatures();

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


  // ---- SO3type --------------------------------------------------------------------------------------------


  py::class_<SO3type>(m,"SO3type","Class to store the type of an SO3-vector")

    .def(pybind11::init<>(),"")
    .def(pybind11::init<vector<int> >(),"")
    
    .def("__len__",&SO3type::size)
    .def("maxl",&SO3type::maxl)
    .def("__getitem__",&SO3type::operator())
    .def("__setitem__",&SO3type::set)
    

    .def("str",&SO3type::str,py::arg("indent")="","Print the SO3type to string.")
    .def("__str__",&SO3type::str,py::arg("indent")="","Print the SO3type to string.")
    .def("__repr__",&SO3type::repr,py::arg("indent")="","Print the SO3type to string.");


  m.def("CGproduct",static_cast<SO3type (*)(const SO3type&, const SO3type&, const int)>(&CGproduct),
    py::arg("x"),py::arg("y"),py::arg("maxl")=-1);

  /*
  m.def("CGproduct",[](const vector<int>& x, const vector<int>& y, const int maxl){ // this causes problems with dispatch
      return CGproduct(SO3type(x),SO3type(y),maxl);},
    py::arg("x"),py::arg("y"),py::arg("maxl")=-1);
  m.def("CGproduct",[](const SO3type& x, const vector<int>& y, const int maxl){
      return CGproduct(x,SO3type(y),maxl);},
    py::arg("x"),py::arg("y"),py::arg("maxl")=-1);
  m.def("CGproduct",[](const vector<int>& x, const SO3type& y, const int maxl){
      return CGproduct(SO3type(x),y,maxl);},
    py::arg("x"),py::arg("y"),py::arg("maxl")=-1);
  */

  //#include "SO3part_py.cpp"
  //#include "SO3vec_py.cpp"
  //  #include "SO3partArray_py.cpp"
  //#include "SO3vecArray_py.cpp"

  #include "SO3partB_py.cpp"
  #include "SO3vecB_py.cpp"
  #include "SO3Fvec_py.cpp"

  #include "SO3partD_py.cpp"
  #include "SO3vecD_py.cpp"

}

