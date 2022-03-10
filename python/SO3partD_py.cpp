
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3partD>(m,"SO3partD",
  "Class to store an array consisting of n vectors transforming according to a specific irreducible representation of SO(3)")
    
//.def_static("raw",static_cast<SO3partD(*)(const int, const int, const int)>(&SO3part::raw))
//.def_static("raw",[](const int b, const int l, const int n, const int dev){
//    return SO3part::raw(b,l,n,dev);}, 
//  py::arg("b"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)

//.def_static("zero",static_cast<SO3partD(*)(const int, const int, const int)>(&SO3partD::zero))
  .def_static("zero",[](const int N, const int b, const int l, const int n, const int dev){
      return SO3partD::zero(b,l,n,dev);}, 
     py::arg("N"), py::arg("b"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)

//.def_static("gaussian",static_cast<SO3partD(*)(const int, const int, const int)>(&SO3partD::gaussian))
  .def_static("gaussian",[](const int N, const int b, const int l, const int n, const int dev){
      return SO3partD::gaussian(N, b,l,n,dev);}, 
    py::arg("N"),  py::arg("b"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)

//.def_static("spharm",[](const int l, const vector<float> v){
//    return SO3part::spharm(l,1,cnine::Gtensor<float>(v));})
//  .def_static("spharm",[](const int l, const cnine::RtensorObj v){
//    return SO3part::spharm(l,1,v.gtensor());})

//.def(pybind11::init([](const at::Tensor& x){return SO3partD(cnine::CtensorB(x));}))
  .def_static("view",[](at::Tensor& x){return SO3partD(cnine::CtensorB::view(x));})
//.def("torch",&cnine::CtensorObj::torch)
  .def("torch",[](const SO3partD& x){return x.torch();})

  .def("__len__",[](const SO3partD& obj){cout<<"111"<<endl;return 1;})

  .def("getN",&SO3partD::getN)
  .def("getl",&SO3partD::getl)
  .def("getn",&SO3partD::getn)

//  .def("__call__",[](const SO3partD& obj, const int i, const int m){return obj.get_value(i,m);})
//  .def("__getitem__",[](const SO3partD& obj, const vector<int> v){
//      return obj.get_value(v[0],v[1]);})
//  .def("__setitem__",[](SO3part& obj, const vector<int> v, const complex<float> x){
//      obj.set_value(v[0],v[1],x);})

  .def("apply",&SO3partD::rotate)

  .def("gather",&SO3partD::add_gather,py::arg("x"),py::arg("mask"))

  .def("device",&SO3partD::get_device)
  .def("to",&SO3partD::to_device)
  .def("to_device",&SO3partD::to_device)
  .def("move_to",[](SO3partD& x, const int _dev){x.move_to_device(_dev);})
    
  .def("str",&SO3partD::str,py::arg("indent")="")
  .def("__str__",&SO3partD::str,py::arg("indent")="")
//.def("__repr__",&SO3partD::repr,py::arg("indent")="")
;


// ---- Stand-alone functions --------------------------------------------------------------------------------
    
m.def("CGproduct",[](const SO3partD& x, const SO3partD& y, const int l){return x.CGproduct(y,l);});

