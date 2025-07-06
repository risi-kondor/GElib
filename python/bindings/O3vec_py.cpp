/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

py::class_<O3vec<float> >(m,"O3vec",
  "Class to store an O(3)-covariant vector")

  .def_static("view",[](map<pair<int,int>,at::Tensor>& parts){
      O3vec<float> R;
      for(auto p:parts)
	R.parts.emplace(O3index(p.first),O3part<float>(tensorc::view(p.second),p.first.second));
      if(R.parts.size()>0){
	auto& P=R.parts.begin()->second;
	R._nbatch=P.getb();
	R._gdims=P.get_gdims();
	R.dev=P.dev;
      }
      return R;
    })

/*
  .def_static("view",[](vector<at::Tensor>& parts){
      O3vec<float> R;
      for(auto p:parts){
	auto P=tensorc::view(p);
	R.parts.emplace((P.dim(-2)-1)/2,O3part<float>(P));
      }
      if(R.parts.size()>0){
	auto& P=R.parts.begin()->second;
	R._nbatch=P.getb();
	R._gdims=P.get_gdims();
	R.dev=P.dev;
      }
      return R;
    })
*/

  .def("get_tau",&O3vec<float>::get_tau)

  .def("addCGproduct",&O3vec<float>::add_CGproduct,py::arg("x"),py::arg("y"))
  .def("addCGproduct_back0",&O3vec<float>::add_CGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addCGproduct_back1",&O3vec<float>::add_CGproduct_back1,py::arg("g"),py::arg("x"))

  .def("addDiagCGproduct",&O3vec<float>::add_DiagCGproduct,py::arg("x"),py::arg("y"))
  .def("addDiagCGproduct_back0",&O3vec<float>::add_DiagCGproduct_back0,py::arg("g"),py::arg("y"))
  .def("addDiagCGproduct_back1",&O3vec<float>::add_DiagCGproduct_back1,py::arg("g"),py::arg("x"))

  .def("str",&O3vec<float>::str,py::arg("indent")="")
  .def("__str__",&O3vec<float>::str,py::arg("indent")="")
  .def("__repr__",&O3vec<float>::repr)
;

