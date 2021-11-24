
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3part>(m,"SO3part",
  "Class to store an array consisting of n vectors transforming according to a specific irreducible representation of SO(3)")
    
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


// ---- Stand-alone functions --------------------------------------------------------------------------------
    
m.def("inp",[](const SO3part& x, const SO3part& y){return x.inp(y);});
//m.def("odot",[](const CtensorObj& x, const CtensorObj& y){return x.odot(y);});
m.def("norm2",[](const SO3part& x){return x.norm2();});

m.def("inp",[](const SO3part& x, const SO3part& y){return x.inp(y);});

m.def("CGproduct",[](const SO3part& x, const SO3part& y, const int l){return CGproduct(x,y,l);});

