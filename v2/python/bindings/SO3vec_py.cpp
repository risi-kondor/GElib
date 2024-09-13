
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3vec<float> >(m,"SO3vec",
  "Class to store an SO(3)-covariant vector")

  .def_static("view",[](map<int,at::Tensor>& parts){
      SO3vec<float> R;
      for(auto p:parts)
	R.parts.emplace(p.first,SO3part<float>(tensorc::view(p.second)));
      if(R.parts.size()>0){
	auto& P=R.parts.begin()->second;
	R._nbatch=P.getb();
	R._gdims=P.get_gdims();
	R.dev=P.dev;
      }
      return R;
    })

  .def_static("view",[](vector<at::Tensor>& parts){
      SO3vec<float> R;
      for(auto p:parts){
	auto P=tensorc::view(p);
	R.parts.emplace((P.dim(-2)-1)/2,SO3part<float>(P));
      }
      if(R.parts.size()>0){
	auto& P=R.parts.begin()->second;
	R._nbatch=P.getb();
	R._gdims=P.get_gdims();
	R.dev=P.dev;
      }
      return R;
    })

  .def("get_tau",&SO3vec<float>::get_tau)

  .def("addCGproduct",&SO3vec<float>::add_CGproduct,py::arg("x"),py::arg("y"))
  .def("addCGproduct_back0",&SO3vec<float>::add_CGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addCGproduct_back1",&SO3vec<float>::add_CGproduct_back1,py::arg("g"),py::arg("x"))

  .def("addDiagCGproduct",&SO3vec<float>::add_DiagCGproduct,py::arg("x"),py::arg("y"))
  .def("addDiagCGproduct_back0",&SO3vec<float>::add_DiagCGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addDiagCGproduct_back1",&SO3vec<float>::add_DiagCGproduct_back1,py::arg("g"),py::arg("x"))

  .def("str",&SO3vec<float>::str,py::arg("indent")="")
  .def("__str__",&SO3vec<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3vec<float>::repr)
;

