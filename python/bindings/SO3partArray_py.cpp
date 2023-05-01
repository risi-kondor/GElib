
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3partArray<float> >(m,"SO3partArray",
  "Class to store an array consisting of n vectors transforming according to a specific irreducible representation of SO(3)")

    
  .def_static("raw",[](const int b, const vector<int>& adims, const int l, const int n, const int dev){
      return SO3partArray<float>::raw(b,adims,l,n,dev);}, 
    py::arg("b"), py::arg("adims"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)
  .def_static("zero",[](const int b, const vector<int>& adims, const int l, const int n, const int dev){
      return SO3partArray<float>::zero(b,adims,l,n,dev);}, 
    py::arg("b"), py::arg("adims"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)
  .def_static("gaussian",[](const int b, const vector<int>& adims, const int l, const int n, const int dev){
      return SO3partArray<float>::gaussian(b,adims,l,n,dev);}, 
    py::arg("b"), py::arg("adims"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)

//.def(pybind11::init([](const at::Tensor& x){return SO3partArray<float>(cnine::CtensorB(x));}))
  .def(pybind11::init<const at::Tensor&>())
  .def("torch",[](const SO3partArray<float>& x){return x.torch();})

  .def("add_to_grad",[](SO3partArray<float>& r, const SO3partArray<float>& x){r.add_to_grad(x);})
  .def("get_grad",&SO3partArray<float>::get_grad)

  .def("__len__",[](const SO3partArray<float>& obj){return 1;})
  .def("device",&SO3partArray<float>::device)
  .def("getb",&SO3partArray<float>::getb)
  //.def("get_adims",[](const SO3partArray<float>& x){return vector<int>(x.get_adims());})
  .def("get_adims",&SO3partArray<float>::get_adims) 
  .def("getl",&SO3partArray<float>::getl)
  .def("getn",&SO3partArray<float>::getn)

  .def("batch",[](SO3partArray<float>& r, int b){return r.batch(b);})
  .def("get_batch_back",[](SO3partArray<float>& r, int b, SO3partArray<float>& x){
      r.get_grad().batch(b).add(x.get_grad());})

  .def("cell",[](SO3partArray<float>& r, vector<int>& ix){return SO3part<float>(r.cell(ix));})
  .def("get_cell_back",[](SO3partArray<float>& r, vector<int>& ix, SO3part<float>& x){
      r.get_grad()(ix).add(x.get_grad());})

//.def("mprod",&SO3partArray<float>B::mprod)
//  .def("add_mprod",&SO3partArray<float>B::add_mprod)
//  .def("add_mprod_back0",&SO3partArray<float>B::add_mprod_back0)
//  .def("add_mprod_back1",&SO3partArray<float>B::add_mprod_back1)

/*
  .def("mprod",static_cast<SO3partArray<float>(SO3partArray<float>::*)(const cnine::CtensorB&)>(&SO3partArray<float>::mprod))
  .def("add_mprod",static_cast<void(SO3partArray<float>::*)(const SO3partArray<float>&, const cnine::CtensorB&)>(&SO3partArray<float>::add_mprod))
  .def("add_mprod_back0",static_cast<void(SO3partArray<float>::*)(const SO3partArray<float>&, const cnine::CtensorB&)>(&SO3partArray<float>::add_mprod_back0))
  .def("add_mprod_back1_into",static_cast<void(SO3partArray<float>::*)(cnine::CtensorB&, const SO3partArray<float>&) const>(&SO3partArray<float>::add_mprod_back1_into))

  .def("add_spharm",[](SO3partArray<float>& obj, const float x, const float y, const float z){
    obj.add_spharm(x,y,z);})
  .def("add_spharm",[](SO3partArray<float>& obj, at::Tensor& _X){
      RtensorA X=RtensorA::view(_X);
      obj.add_spharm(X);})
  .def("add_spharmB",[](SO3partArray<float>& obj, at::Tensor& _X){
      RtensorA X=RtensorA::view(_X);
      obj.add_spharmB(X);})
*/

  .def("add_CGproduct",[](SO3partArray<float>& r, const SO3partArray<float>& x, const SO3partArray<float>& y, const int offs){
      r.add_CGproduct(x,y,offs);},py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("add_CGproduct_back0",[](SO3partArray<float>& r, SO3partArray<float>& g, const SO3partArray<float>& y, const int offs){
      r.get_grad().add_CGproduct_back0(g.get_grad(),y,offs);},py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("add_CGproduct_back1",[](SO3partArray<float>& r, SO3partArray<float>& g, const SO3partArray<float>& x, const int offs){
      r.get_grad().add_CGproduct_back1(g.get_grad(),x,offs);},py::arg("g"),py::arg("x"),py::arg("offs")=0)

  .def("add_DiagCGproduct",[](SO3partArray<float>& r, const SO3partArray<float>& x, const SO3partArray<float>& y, const int offs){
      r.add_DiagCGproduct(x,y,offs);},py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("add_DiagCGproduct_back0",[](SO3partArray<float>& r, SO3partArray<float>& g, const SO3partArray<float>& y, const int offs){
      r.get_grad().add_DiagCGproduct_back0(g.get_grad(),y,offs);},py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("add_DiagCGproduct_back1",[](SO3partArray<float>& r, SO3partArray<float>& g, const SO3partArray<float>& x, const int offs){
      r.get_grad().add_DiagCGproduct_back1(g.get_grad(),x,offs);},py::arg("g"),py::arg("x"),py::arg("offs")=0)



/*
  .def("apply",&SO3partArray<float>::rotate)
  .def("to",&SO3partArray<float>::to_device)
  .def("to_device",&SO3partArray<float>::to_device)
  .def("move_to",[](SO3partArray<float>& x, const int _dev){x.move_to_device(_dev);})
*/
    
  .def("str",&SO3partArray<float>::str,py::arg("indent")="")
  .def("__str__",&SO3partArray<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3partArray<float>::repr,py::arg("indent")="")
;


// ---- Stand-alone functions --------------------------------------------------------------------------------

    
m.def("CGproduct",[](const SO3partArray<float>& x, const SO3partArray<float>& y, const int maxl){
    return CGproduct(x,y,maxl);});


