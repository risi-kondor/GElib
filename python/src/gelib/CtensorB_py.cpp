
//  This file is part of cnine, a lightweight C++ tensor library. 
// 
//  Copyright (c) 2021, Imre Risi Kondor
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


pybind11::class_<CtensorB>(m,"ctensorb")

  .def(pybind11::init<const Gdims&>())
  .def(pybind11::init<const Gdims&, const fill_raw&>())
  .def(pybind11::init<const Gdims&, const fill_zero&>())
  .def(pybind11::init<const Gdims&, const fill_ones&>())
//.def(pybind11::init<const Gdims&, const fill_identity&>())
//.def(pybind11::init<const Gdims&, const fill_gaussian&>())
//.def(pybind11::init<const Gdims&, const fill_sequential&>())

  .def(pybind11::init<const at::Tensor&>())
  .def_static("view",static_cast<CtensorB(*)(at::Tensor&)>(&CtensorB::view))
//  .def_static("is_viewable",static_cast<bool(*)(const at::Tensor&)>(&CtensorB::is_viewable))
//.def_static("view",static_cast<CtensorObj>(*)(const at::Tensor&)>(&CtensorObj::view))
//.def_static("const_view",static_cast<CtensorObj>(*)(at::Tensor&)>(&CtensorObj::const_view))
  .def("torch",&CtensorB::torch)

  .def("get_dim",&CtensorB::get_dim)
  .def("get_dev",&CtensorB::get_dev)

  .def("str",&CtensorB::str,py::arg("indent")="")
  .def("__str__",&CtensorB::str,py::arg("indent")="")
  .def("__repr__",&CtensorB::str,py::arg("indent")="");

