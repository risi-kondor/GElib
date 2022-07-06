
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


pybind11::class_<SO3partB_array>(m,"SO3partB_array",
  "Class to store an array of SO3part objects.")

  .def_static("zero",[](const Gdims& adims, const int l, const int n, const int dev){
      return SO3partB_array::zero(adims,l,n,dev);}, 
    py::arg("adims"), py::arg("l"), py::arg("n"), py::arg("device")=0)
  .def_static("zero",[](const vector<int>& av, const int l, int n, const int dev){
      return SO3partB_array::zero(Gdims(av),l,n,dev);},
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

  .def_static("ones",[](const Gdims& adims, const int l, const int n, const int dev){
      return SO3partB_array::ones(adims,l,n,dev);}, 
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)
  .def_static("ones",[](const vector<int>& av, const int l, const int n, const int dev){
      return SO3partB_array::ones(Gdims(av),l,n,dev);},
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

  .def_static("gaussian",[](const Gdims& adims, const int l, const int n, const int dev){
      return SO3partB_array::gaussian(adims,l,n,dev);}, 
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<int>& av, const int l, const int n, const int dev){
      return SO3partB_array::gaussian(Gdims(av),l,n,dev);},
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

//.def_static("view",[](at::Tensor& x){return SO3partB_array(cnine::CtensorB::view(x));})
  .def_static("view",[](at::Tensor& x){return SO3partB_array(SO3partB_array::view(x,-2));})
  .def("torch",[](const SO3partB_array& x){return x.torch();})

  .def("get_adims",[](const SO3partB_array& x){return vector<int>(x.get_adims());})
  .def("getl",&SO3partB_array::getl)
  .def("getn",&SO3partB_array::getn)

  .def("get_adims",&SO3partB_array::get_adims)

  .def("get_cell",[](const SO3partB_array& obj, const Gindex& ix){
      return SO3partB(obj.get_cell(ix));})
  .def("get_cell",[](const SO3partB_array& obj, const vector<int> v){
      return SO3partB(obj.get_cell(Gindex(v)));})
  .def("__call__",[](const SO3partB_array& obj, const Gindex& ix){
      return SO3partB(obj.get_cell(ix));})
  .def("__call__",[](const SO3partB_array& obj, const vector<int> v){
      return SO3partB(obj.get_cell(Gindex(v)));})
  .def("__getitem__",[](const SO3partB_array& obj, const Gindex& ix){
      return SO3partB(obj.get_cell(ix));})
  .def("__getitem__",[](const SO3partB_array& obj, const vector<int> v){
      return obj.get_cell(Gindex(v));})

  .def("__iadd__",[](SO3partB_array& x, const SO3partB_array& y){x.add(y); return x;})
  .def("__isub__",[](SO3partB_array& x, const SO3partB_array& y){x.subtract(y); return x;})

//.def("widen",&SO3partB_array::widen)
//.def("reduce",&SO3partB_array::reduce)

//.def("apply",&SO3partB_array::rotate)
  .def("rotate",[](const SO3partB_array& x, const SO3element& R){return SO3partB_array(x.rotate(R));})

  .def("gather",&SO3partB_array::add_gather,py::arg("x"),py::arg("mask"))

  .def("addCGproduct",&SO3partB_array::add_CGproduct,py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("addCGproduct_back0",&SO3partB_array::add_CGproduct_back0,py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("addCGproduct_back1",&SO3partB_array::add_CGproduct_back1,py::arg("g"),py::arg("x"),py::arg("offs")=0)

  .def("addDiagCGproduct",&SO3partB_array::add_DiagCGproduct,py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("addDiagCGproduct_back0",&SO3partB_array::add_DiagCGproduct_back0,py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("addDiagCGproduct_back1",&SO3partB_array::add_DiagCGproduct_back1,py::arg("g"),py::arg("x"),py::arg("offs")=0)

  .def("device",&SO3partB_array::get_device)
  .def("to",&SO3partB_array::to_device)
  .def("to_device",&SO3partB_array::to_device)
  .def("move_to",[](SO3partB_array& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&SO3partB_array::str,py::arg("indent")="")
  .def("__str__",&SO3partB_array::str,py::arg("indent")="")
  .def("__repr__",&SO3partB_array::repr,py::arg("indent")="");


// ---- Stand-alone functions --------------------------------------------------------------------------------



