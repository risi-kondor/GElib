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


py::class_<O3type>(m,"O3type","Class to store the type of an O3-vector")

  .def(pybind11::init<>(),"")
//.def(pybind11::init<vector<int> >(),"")
  .def(pybind11::init<map<pair<int,int>,int> >(),"")
    
  .def("__len__",&O3type::size)
  .def("maxl",&O3type::highest)
  .def("__getitem__",&O3type::operator())
  .def("__setitem__",&O3type::set)
  .def("get_parts",[](const O3type& x){return x.parts;})
    
  .def("CGproduct",[](const O3type& x, const O3type& y) {return CGproduct(x,y,-1);})
  .def("CGproduct",[](const O3type& x, const O3type& y, int maxl) {return CGproduct(x,y,maxl);})

  .def("DiagCGproduct",[](const O3type& x, const O3type& y) {return x.DiagCGproduct(y,-1);})
  .def("DiagCGproduct",[](const O3type& x, const O3type& y, int maxl) {return x.DiagCGproduct(y,maxl);})

  .def("str",&O3type::str,py::arg("indent")="","Print the O3type to string.")
  .def("__str__",&O3type::str,py::arg("indent")="","Print the O3type to string.")
  .def("__repr__",&O3type::repr,"Print the O3type to string.");


m.def("CGproduct",[](const O3type& x, const O3type& y){
    return CGproduct(x,y,-1);});

m.def("CGproduct",[](const O3type& x, const O3type& y, const int maxl){
    return CGproduct(x,y,maxl);});


