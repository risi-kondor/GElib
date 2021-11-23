
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "GElib_base.cpp"
#include "GElibSession.hpp"

#include "RtensorObj.hpp"

#include "SO3type.hpp"
#include "SO3part.hpp"
#include "SO3vec.hpp"


//std::default_random_engine rndGen;

GElib::GElibSession session;


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


  using namespace cnine;
  using namespace GElib;
  namespace py=pybind11;

  py::options options;
  //options.disable_function_signatures();


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


  // ---- SO3part --------------------------------------------------------------------------------------------


  py::class_<SO3part>(m,"SO3part",
    "Class to store n vectors transforming according to the same irreducible represenation D^l of SO(3)")

    .def_static("zero",static_cast<SO3part (*)(const int, const int)>(&SO3part::zero))
    .def_static("ones",static_cast<SO3part (*)(const int, const int)>(&SO3part::ones))
    .def_static("gaussian",static_cast<SO3part (*)(const int, const int)>(&SO3part::gaussian))

    .def_static("spharm",[](const int l, const vector<float> v){
	return SO3part::spharm(l,1,cnine::Gtensor<float>(v));})
    .def_static("spharm",[](const int l, const cnine::RtensorObj v){
	return SO3part::spharm(l,1,v.gtensor());})

    .def("getl",&SO3part::getl)
    .def("getn",&SO3part::getn)

    .def("__call__",[](const SO3part& obj, const int i, const int m){return obj.get_value(i,m);})
    .def("__getitem__",[](const SO3part& obj, const vector<int> v){
	return obj.get_value(v[0],v[1]);})
    .def("__setitem__",[](SO3part& obj, const vector<int> v, const complex<float> x){
	obj.set_value(v[0],v[1],x);})

    .def("__add__",[](const SO3part& x, const SO3part& y){return x+y;})
    .def("__sub__",[](const SO3part& x, const SO3part& y){return x-y;})
    .def("__mul__",[](const SO3part& x, const float c){
	SO3part R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
	R.add(x,c);
	return R;})
    .def("__rmul__",[](const SO3part& x, const float c){
	SO3part R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
	R.add(x,c);
	return R;})
    
    .def("__mul__",[](const SO3part& x, const cnine::CtensorObj& M){
	return x*M;})

    .def("__iadd__",[](SO3part& x, const SO3part& y){x+=y; return x;})
    .def("__isub__",[](SO3part& x, const SO3part& y){x+=y; return x;})
    
    .def("to",&SO3part::to_device)
    
    .def("str",&SO3part::str,py::arg("indent")="")
    .def("__str__",&SO3part::str,py::arg("indent")="")
    .def("__repr__",&SO3part::repr,py::arg("indent")="");
    
  m.def("inp",[](const SO3part& x, const SO3part& y){return x.inp(y);});
  //m.def("odot",[](const CtensorObj& x, const CtensorObj& y){return x.odot(y);});
  m.def("norm2",[](const SO3part& x){return x.norm2();});

  m.def("inp",[](const SO3part& x, const SO3part& y){return x.inp(y);});

  m.def("CGproduct",[](const SO3part& x, const SO3part& y, const int l){return CGproduct(x,y,l);});


#include "SO3vec_py.cpp"

}

