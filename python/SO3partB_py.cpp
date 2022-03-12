
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3partB>(m,"SO3partB",
  "Class to store an array consisting of n vectors transforming according to a specific irreducible representation of SO(3)")
    
//.def_static("raw",static_cast<SO3partB(*)(const int, const int, const int)>(&SO3part::raw))
//.def_static("raw",[](const int b, const int l, const int n, const int dev){
//    return SO3part::raw(b,l,n,dev);}, 
//  py::arg("b"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)

//.def_static("zero",static_cast<SO3partB(*)(const int, const int, const int)>(&SO3partB::zero))
  .def_static("zero",[](const int b, const int l, const int n, const int dev){
      return SO3partB::zero(b,l,n,dev);}, 
    py::arg("b"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)

//.def_static("gaussian",static_cast<SO3partB(*)(const int, const int, const int)>(&SO3partB::gaussian))
  .def_static("gaussian",[](const int b, const int l, const int n, const int dev){
      return SO3partB::gaussian(b,l,n,dev);}, 
    py::arg("b"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)

//  .def_static("spharm",[](const int l, const vector<float> v){
//      return SO3partB::spharm(l,1,cnine::Gtensor<float>(v));})
//  .def_static("spharm",[](const int l, const cnine::RtensorObj v){
//    return SO3part::spharm(l,1,v.gtensor());})

//.def(pybind11::init([](const at::Tensor& x){return SO3partB(cnine::CtensorB(x));}))
  .def_static("view",[](at::Tensor& x){return SO3partB(cnine::CtensorB::view(x));})
//.def("torch",&cnine::CtensorObj::torch)
  .def("torch",[](const SO3partB& x){return x.torch();})

  .def("__len__",[](const SO3partB& obj){cout<<"111"<<endl;return 1;})

  .def("getl",&SO3partB::getl)
  .def("getn",&SO3partB::getn)

//  .def("__call__",[](const SO3partB& obj, const int i, const int m){return obj.get_value(i,m);})
//  .def("__getitem__",[](const SO3partB& obj, const vector<int> v){
//      return obj.get_value(v[0],v[1]);})
//  .def("__setitem__",[](SO3part& obj, const vector<int> v, const complex<float> x){
//      obj.set_value(v[0],v[1],x);})

  .def("add_spharm",&SO3partB::add_spharm)

  .def("apply",&SO3partB::rotate)

  .def("device",&SO3partB::get_device)
  .def("to",&SO3partB::to_device)
  .def("to_device",&SO3partB::to_device)
  .def("move_to",[](SO3partB& x, const int _dev){x.move_to_device(_dev);})
    
  .def("str",&SO3partB::str,py::arg("indent")="")
  .def("__str__",&SO3partB::str,py::arg("indent")="")
//.def("__repr__",&SO3partB::repr,py::arg("indent")="")
;


// ---- Stand-alone functions --------------------------------------------------------------------------------
    
m.def("CGproduct",[](const SO3partB& x, const SO3partB& y, const int l){return x.CGproduct(y,l);});

