/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

pybind11::class_<RtensorArray>(m,"rtensorArr")

  .def(pybind11::init<const Gdims&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_raw&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_zero&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_ones&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_identity&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_gaussian&>())
  .def(pybind11::init<const Gdims&, const Gdims&, const fill_sequential&>())

  .def(pybind11::init<const int, const at::Tensor&>())
  .def("torch",&RtensorArray::torch)
  .def_static("is_viewable",static_cast<bool(*)(const at::Tensor&, const int)>(&RtensorArray::is_viewable))

  .def_static("raw",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int)>(&RtensorArray::raw))
  .def_static("raw",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::raw(dims,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("raw",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::raw(Gdims(av),Gdims(v),dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("zero",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int)>(&RtensorArray::zero))
  .def_static("zero",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::zero(dims,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("zero",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::zero(Gdims(av),Gdims(v),dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("zeros",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int)>(&RtensorArray::zero))
  .def_static("zeros",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::zero(dims,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("zeros",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::zero(Gdims(av),Gdims(v),dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("ones",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int)>(&RtensorArray::ones))
  .def_static("ones",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::ones(dims,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("ones",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::ones(Gdims(av),Gdims(v),dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("sequential",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int)>(&RtensorArray::sequential))
  .def_static("sequential",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::sequential(dims,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("sequential",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::sequential(Gdims(av),Gdims(v),dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("gaussian",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int)>(&RtensorArray::gaussian))
  .def_static("gaussian",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::gaussian(dims,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("gaussian",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::gaussian(Gdims(av),Gdims(v),dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

  .def_static("randn",static_cast<RtensorArray (*)(const Gdims&, const Gdims&, const int)>(&RtensorArray::gaussian))
  .def_static("randn",[](const Gdims& adims, const Gdims& dims, const int dev){
      return RtensorArray::gaussian(dims,dev);}, py::arg("adims"), py::arg("dims"), py::arg("device")=0)
  .def_static("randn",[](const vector<int>& av, const vector<int>& v, const int dev){
      return RtensorArray::gaussian(Gdims(av),Gdims(v),dev);},py::arg("adims"),py::arg("dims"),py::arg("device")=0)

//.def("get_k",&RtensorObj::get_k)
//    .def("getk",&RtensorObj::get_k)
  
  .def("get_nadims",&RtensorArray::get_nadims)
  .def("get_adims",&RtensorArray::get_adims)
  .def("get_adim",&RtensorArray::get_adim)
  .def("nadims",&RtensorArray::get_nadims)
  .def("adims",&RtensorArray::get_adims)
  .def("adim",&RtensorArray::get_adim)

  .def("get_ncdims",&RtensorArray::get_cdims)
  .def("get_cdims",&RtensorArray::get_cdims)
  .def("get_cdim",&RtensorArray::get_cdim)
  .def("ncdims",&RtensorArray::get_cdims)
  .def("cdims",&RtensorArray::get_cdims)
  .def("cdim",&RtensorArray::get_cdim)

  .def("get_cell",&RtensorArray::get_cell)
  .def("get_cell",[](const RtensorArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})
  .def("__call__",&RtensorArray::get_cell)
  .def("__call__",[](const RtensorArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})
  .def("__getitem__",&RtensorArray::get_cell)
  .def("__getitem__",[](const RtensorArray& obj, const vector<int> v){return obj.get_cell(Gindex(v));})

  .def("__setitem__",[](RtensorArray& obj, const Gindex& ix, const RtensorObj& x){
      obj.set_cell(ix,x);})
  .def("__setitem__",[](RtensorArray& obj, const vector<int> v, const RtensorObj& x){
      obj.set_cell(Gindex(v),x);})

  .def("__add__",[](const RtensorArray& x, const RtensorArray& y){return x.plus(y);})
  .def("__sub__",[](const RtensorArray& x, const RtensorArray& y){return x-y;})
  .def("__mul__",[](const RtensorArray& x, const float c){
      RtensorArray R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__rmul__",[](const RtensorArray& x, const float c){
      RtensorArray R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;})
  .def("__mul__",[](const RtensorArray& x, const RtensorArray& y){
      RtensorArray R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
      R.add_mprod(x,y);
      return R;})

  .def("__iadd__",[](RtensorArray& x, const RtensorArray& y){x.add(y); return x;})
  .def("__isub__",[](RtensorArray& x, const RtensorArray& y){x.subtract(y); return x;})

  .def("__add__",[](const RtensorArray& x, const RtensorObj& y){return x.broadcast_plus(y);})
  .def("__add__",[](const RtensorObj& y, const RtensorArray& x){return x.broadcast_plus(y);})
  .def("__sub__",[](const RtensorArray& x, const RtensorObj& y){return x.broadcast_minus(y);})
//.def("__sub__",[](const RtensorObj& y, const RtensorArray& x){return x.broadcast_minus(y);})

  .def("widen",&RtensorArray::widen)
  .def("reduce",&RtensorArray::reduce)

  .def("device",&RtensorArray::get_device)
  .def("to",&RtensorArray::to_device)
  .def("to_device",&RtensorArray::to_device)
  .def("move_to",[](RtensorArray& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&RtensorArray::str,py::arg("indent")="")
  .def("__str__",&RtensorArray::str,py::arg("indent")="")
  .def("__repr__",&RtensorArray::repr);
