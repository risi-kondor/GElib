
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
    
  .def_static("raw",static_cast<SO3part (*)(const int, const int)>(&SO3part::raw))
  .def_static("raw",[](const int l, const int n, const int dev) {return SO3part::raw(l,n,-1,dev);}, 
    py::arg("l"), py::arg("n")=1, py::arg("device")=0)

  .def_static("zero",static_cast<SO3part (*)(const int, const int)>(&SO3part::zero))
  .def_static("zero",[](const int l, const int n, const int dev) {return SO3part::zero(l,n,-1,dev);}, 
    py::arg("l"), py::arg("n")=1, py::arg("device")=0)

  .def_static("ones",static_cast<SO3part (*)(const int, const int)>(&SO3part::ones))
  .def_static("ones",[](const int l, const int n, const int dev) {return SO3part::ones(l,n,-1,dev);}, 
    py::arg("l"), py::arg("n")=1, py::arg("device")=0)

  .def_static("gaussian",static_cast<SO3part (*)(const int, const int)>(&SO3part::gaussian))
  .def_static("gaussian",[](const int l, const int n, const int dev) {return SO3part::gaussian(l,n,-1,dev);}, 
    py::arg("l"), py::arg("n")=1, py::arg("device")=0)

  .def_static("spharm",[](const int l, const vector<float> v){
      return SO3part::spharm(l,1,cnine::Gtensor<float>(v));})
  .def_static("spharm",[](const int l, const cnine::RtensorObj v){
      return SO3part::spharm(l,1,v.gtensor());})

  .def(pybind11::init([](const at::Tensor& x){return SO3part(cnine::CtensorObj(x));}))
  .def_static("view",[](at::Tensor& x){return SO3part(cnine::CtensorObj::view(x));})
//.def("torch",&cnine::CtensorObj::torch)
  .def("torch",[](const SO3part& x){return x.torch();})

  .def("__len__",[](const SO3part& obj){cout<<"111"<<endl;return 1;})

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

  .def("apply",&SO3part::rotate)

  .def("addFourierConjugate",&SO3part::add_FourierConjugate)
  
  .def("addFullCGproduct",&SO3part::add_CGproduct,py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("addFullCGproduct_back0",&SO3part::add_CGproduct_back0,py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("addFullCGproduct_back1",&SO3part::add_CGproduct_back1,py::arg("g"),py::arg("x"),py::arg("offs")=0)

  .def("addDiagCGproduct",&SO3part::add_DiagCGproduct,py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("addDiagCGproduct_back0",&SO3part::add_DiagCGproduct_back0,py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("addDiagCGproduct_back1",&SO3part::add_DiagCGproduct_back1,py::arg("g"),py::arg("x"),py::arg("offs")=0)

  .def("addBlockwiseCGproduct",&SO3part::add_BlockwiseCGproduct,py::arg("x"),py::arg("y"),py::arg("nblocks"),py::arg("offs")=0)
  .def("addBlockwiseCGproduct_back0",&SO3part::add_BlockwiseCGproduct_back0,py::arg("g"),py::arg("y"),py::arg("nblocks"),py::arg("offs")=0)
  .def("addBlockwiseCGproduct_back1",&SO3part::add_BlockwiseCGproduct_back1,py::arg("g"),py::arg("x"),py::arg("nblocks"),py::arg("offs")=0)

  .def("device",&SO3part::get_device)
  .def("to",&SO3part::to_device)
  .def("to_device",&SO3part::to_device)
  .def("move_to",[](SO3part& x, const int _dev){x.move_to_device(_dev);})
    
  .def("str",&SO3part::str,py::arg("indent")="")
  .def("__str__",&SO3part::str,py::arg("indent")="")
  .def("__repr__",&SO3part::repr,py::arg("indent")="");


// ---- Stand-alone functions --------------------------------------------------------------------------------
    
m.def("inp",[](const SO3part& x, const SO3part& y){return x.inp(y);});
//m.def("odot",[](const CtensorObj& x, const CtensorObj& y){return x.odot(y);});
m.def("norm2",[](const SO3part& x){return x.norm2();});

m.def("inp",[](const SO3part& x, const SO3part& y){return x.inp(y);});

m.def("CGproduct",[](const SO3part& x, const SO3part& y, const int l){return CGproduct(x,y,l);});

