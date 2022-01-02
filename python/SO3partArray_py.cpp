
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


pybind11::class_<SO3partArray>(m,"SO3partArr",
  "Class to store an array of SO3part objects.")

//.def_static("zero",static_cast<SO3partArray (*)(const Gdims&, const int, const int)>(&SO3partArray::zero))
  .def_static("zero",[](const Gdims& adims, const int l, const int n, const int dev){
      return SO3partArray::zero(adims,l,n,-1,dev);}, 
    py::arg("adims"), py::arg("l"), py::arg("n"), py::arg("device")=0)
  .def_static("zero",[](const vector<int>& av, const int l, int n, const int dev){
      return SO3partArray::zero(Gdims(av),l,n,-1,dev);},
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

//.def_static("ones",static_cast<SO3partArray (*)(const Gdims&, const Gdims&, const int, const int)>(&SO3partArray::ones))
  .def_static("ones",[](const Gdims& adims, const int l, const int n, const int dev){
      return SO3partArray::ones(adims,l,n,-1,dev);}, 
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)
  .def_static("ones",[](const vector<int>& av, const int l, const int n, const int dev){
      return SO3partArray::ones(Gdims(av),l,n,-1,dev);},
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

//.def_static("gaussian",static_cast<SO3partArray (*)(const Gdims&, const Gdims&, const int, const int)>(&SO3partArray::gaussian))
  .def_static("gaussian",[](const Gdims& adims, const int l, const int n, const int dev){
      return SO3partArray::gaussian(adims,l,n,-1,dev);}, 
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)
  .def_static("gaussian",[](const vector<int>& av, const int l, const int n, const int dev){
      return SO3partArray::gaussian(Gdims(av),l,n,-1,dev);},
    py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

  .def("getl",&SO3partArray::getl)
  .def("getn",&SO3partArray::getn)

  .def("get_adims",&CtensorArray::get_adims)
  .def("get_adim",&CtensorArray::get_adim)

  .def("get_cell",[](const SO3partArray& obj, const Gindex& ix){
      return SO3partArray(obj.get_cell(ix));})
  .def("get_cell",[](const SO3partArray& obj, const vector<int> v){
      return SO3partArray(obj.get_cell(Gindex(v)));})
  .def("__call__",[](const SO3partArray& obj, const Gindex& ix){
      return SO3partArray(obj.get_cell(ix));})
  .def("__call__",[](const SO3partArray& obj, const vector<int> v){
      return SO3partArray(obj.get_cell(Gindex(v)));})
  .def("__getitem__",[](const SO3partArray& obj, const Gindex& ix){
      return SO3partArray(obj.get_cell(ix));})
  .def("__getitem__",[](const SO3partArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})

  .def("__setitem__",[](SO3partArray& obj, const Gindex& ix, const SO3part& x){
      obj.set_cell(ix,x);})
  .def("__setitem__",[](SO3partArray& obj, const vector<int> v, const SO3part& x){
      obj.set_cell(Gindex(v),x);})

  .def("__add__",[](const SO3partArray& x, const SO3partArray& y){
      return SO3partArray(x.plus(y));})
  .def("__sub__",[](const SO3partArray& x, const SO3partArray& y){
      return SO3partArray(x-y);})
  .def("__mul__",[](const SO3partArray& x, const float c){
      SO3partArray R(x.get_adims(),x.getl(),x.getn(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__rmul__",[](const SO3partArray& x, const float c){
      SO3partArray R(x.get_dims(),x.getl(),x.getn(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
//.def("__mul__",[](const SO3partArray& x, const SO3partArray& y){
//    SO3partArray R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
//    R.add_mprod(x,y);
//    return R;})

  .def("__iadd__",[](SO3partArray& x, const SO3partArray& y){x.add(y); return x;})
  .def("__isub__",[](SO3partArray& x, const SO3partArray& y){x.subtract(y); return x;})

//.def("__add__",[](const SO3partArray& x, const SO3part& y){return x.broadcast_plus(y);})

//.def("widen",&SO3partArray::widen)
//.def("reduce",&SO3partArray::reduce)

  .def("device",&SO3partArray::get_device)
  .def("to",&SO3partArray::to_device)
  .def("to_device",&SO3partArray::to_device)
  .def("move_to",[](SO3partArray& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&SO3partArray::str,py::arg("indent")="")
  .def("__str__",&SO3partArray::str,py::arg("indent")="")
  .def("__repr__",&SO3partArray::repr,py::arg("indent")="");


// ---- Stand-alone functions --------------------------------------------------------------------------------


//m.def("inp",[](const SO3partArray& x, const SO3partArray& y){return x.inp(y);});
//m.def("odot",[](const CtensorObj& x, const CtensorObj& y){return x.odot(y);});
//m.def("norm2",[](const SO3partArray& x){return x.norm2();});

//m.def("inp",[](const SO3partArray& x, const SO3partArray& y){return x.inp(y);});

m.def("CGproduct",[](const SO3partArray& x, const SO3partArray& y, const int l){return CGproduct(x,y,l);});

