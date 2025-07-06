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

py::class_<SO3part<float> >(m,"SO3part",
  "Class to store an array consisting of n vectors transforming according to a specific irreducible representation of SO(3)")

  .def_static("view",[](at::Tensor& x){
      return SO3part<float>(tensorc::view(x));})

  .def("add_gather",[](SO3part<float>& r, const SO3part<float>& x, const GatherMapB& gmap, const int d){
      r.add_gather(x,gmap,d);})
  .def("add_gather_back",[](SO3part<float>& xg, const SO3part<float>& rg, const GatherMapB& gmap, const int d){
      xg.add_gather(rg,gmap,d);})

//  .def("add_spharm",[](SO3part<float>& obj, const float x, const float y, const float z){
//    obj.add_spharm(x,y,z);})
  .def("add_spharm",[](SO3part<float>& obj, at::Tensor& X){
      obj.add_spharm(tensorf::view(X));})


  .def("add_CGproduct",[](SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs){
	 Gpart_add_CGproduct(r,x,y,offs);},py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("add_CGproduct_back0",[](SO3part<float>& r, SO3part<float>& g, const SO3part<float>& y, const int offs){
	 Gpart_add_CGproduct_back0(r,g,y,offs);},py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("add_CGproduct_back1",[](SO3part<float>& r, SO3part<float>& g, const SO3part<float>& x, const int offs){
	 Gpart_add_CGproduct_back1(r,g,x,offs);},py::arg("g"),py::arg("x"),py::arg("offs")=0)
  
  .def("add_DiagCGproduct",[](SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs){
	 Gpart_add_DiagCGproduct(r,x,y,offs);},py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("add_DiagCGproduct_back0",[](SO3part<float>& r, SO3part<float>& g, const SO3part<float>& y, const int offs){
	 Gpart_add_DiagCGproduct_back0(r,g,y,offs);},py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("add_DiagCGproduct_back1",[](SO3part<float>& r, SO3part<float>& g, const SO3part<float>& x, const int offs){
	 Gpart_add_DiagCGproduct_back1(r,g,x,offs);},py::arg("g"),py::arg("x"),py::arg("offs")=0)
  
//  .def("apply",&SO3part<float>::rotate)
  
  .def("str",&SO3part<float>::str,py::arg("indent")="")
  .def("__str__",&SO3part<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3part<float>::repr)
;


// ---- Stand-alone functions --------------------------------------------------------------------------------

    
m.def("CGproduct",[](const SO3part<float>& x, const SO3part<float>& y, const int l){
    return CGproduct(x,y,l);});

