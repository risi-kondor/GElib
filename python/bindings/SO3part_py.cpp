
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


py::class_<SO3part<float> >(m,"SO3part",
  "Class to store an array consisting of n vectors transforming according to a specific irreducible representation of SO(3)")

  .def(pybind11::init<const at::Tensor&>())
    
  .def_static("raw",[](const int b, const int l, const int n, const int dev){
    return SO3part<float>::raw(b,l,n,dev);}, py::arg("b"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)
  .def_static("zero",[](const int b, const int l, const int n, const int dev){
      return SO3part<float>::zero(b,l,n,dev);}, py::arg("b"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)
  .def_static("gaussian",[](const int b, const int l, const int n, const int dev){
      return SO3part<float>::gaussian(b,l,n,dev);}, py::arg("b"), py::arg("l"), py::arg("n")=1, py::arg("device")=0)

//.def(pybind11::init([](const at::Tensor& x){return SO3part<float>(cnine::CtensorB(x));}))
//  .def_static("view",[](at::Tensor& x){return SO3part<float>(cnine::CtensorB::view(x));})
  .def("torch",[](const SO3part<float>& x){return x.torch();})

  .def("add_to_grad",[](SO3part<float>& r, const SO3part<float>& x){r.add_to_grad(x);})
  .def("get_grad",[](SO3part<float>& part) { return part.get_grad(); })

  .def("__len__",[](const SO3part<float>& obj){return 1;})
  .def("device",&SO3part<float>::device)
  .def("getb",&SO3part<float>::getb)
  .def("getl",&SO3part<float>::getl)
  .def("getn",&SO3part<float>::getn)

  .def("batch",[](SO3part<float>& r, int b){return r.batch(b);})
  .def("get_batch_back",[](SO3part<float>& r, int b, SO3part<float>& x){
      r.get_grad().batch(b).add(x.get_grad());})

//.def("mprod",&SO3part<float>B::mprod)
//  .def("add_mprod",&SO3part<float>B::add_mprod)
//  .def("add_mprod_back0",&SO3part<float>B::add_mprod_back0)
//  .def("add_mprod_back1",&SO3part<float>B::add_mprod_back1)

/*
  .def("mprod",static_cast<SO3part<float>(SO3part<float>::*)(const cnine::CtensorB&)>(&SO3part<float>::mprod))
  .def("add_mprod",static_cast<void(SO3part<float>::*)(const SO3part<float>&, const cnine::CtensorB&)>(&SO3part<float>::add_mprod))
  .def("add_mprod_back0",static_cast<void(SO3part<float>::*)(const SO3part<float>&, const cnine::CtensorB&)>(&SO3part<float>::add_mprod_back0))
  .def("add_mprod_back1_into",static_cast<void(SO3part<float>::*)(cnine::CtensorB&, const SO3part<float>&) const>(&SO3part<float>::add_mprod_back1_into))

  .def("add_spharm",[](SO3part<float>& obj, const float x, const float y, const float z){
    obj.add_spharm(x,y,z);})
  .def("add_spharm",[](SO3part<float>& obj, at::Tensor& _X){
      RtensorA X=RtensorA::view(_X);
      obj.add_spharm(X);})
  .def("add_spharmB",[](SO3part<float>& obj, at::Tensor& _X){
      RtensorA X=RtensorA::view(_X);
      obj.add_spharmB(X);})
*/

  .def("add_CGproduct",[](SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs){
      r.add_CGproduct(x,y,offs);},py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("add_CGproduct_back0",[](SO3part<float>& r, SO3part<float>& g, const SO3part<float>& y, const int offs){
      r.get_grad().add_CGproduct_back0(g.get_grad(),y,offs);},py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("add_CGproduct_back1",[](SO3part<float>& r, SO3part<float>& g, const SO3part<float>& x, const int offs){
      r.get_grad().add_CGproduct_back1(g.get_grad(),x,offs);},py::arg("g"),py::arg("x"),py::arg("offs")=0)

  .def("add_DiagCGproduct",[](SO3part<float>& r, const SO3part<float>& x, const SO3part<float>& y, const int offs){
      r.add_DiagCGproduct(x,y,offs);},py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("add_DiagCGproduct_back0",[](SO3part<float>& r, SO3part<float>& g, const SO3part<float>& y, const int offs){
      r.get_grad().add_DiagCGproduct_back0(g.get_grad(),y,offs);},py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("add_DiagCGproduct_back1",[](SO3part<float>& r, SO3part<float>& g, const SO3part<float>& x, const int offs){
      r.get_grad().add_DiagCGproduct_back1(g.get_grad(),x,offs);},py::arg("g"),py::arg("x"),py::arg("offs")=0)



/*
  .def("apply",&SO3part<float>::rotate)
  .def("to",&SO3part<float>::to_device)
  .def("to_device",&SO3part<float>::to_device)
  .def("move_to",[](SO3part<float>& x, const int _dev){x.move_to_device(_dev);})
*/
    
  .def("str",&SO3part<float>::str,py::arg("indent")="")
  .def("__str__",&SO3part<float>::str,py::arg("indent")="")
  .def("__repr__",&SO3part<float>::repr,py::arg("indent")="")
;


// ---- Stand-alone functions --------------------------------------------------------------------------------

    
m.def("CGproduct",[](const SO3part<float>& x, const SO3part<float>& y, const int l){
    return CGproduct(x,y,l);});



/*
  .def_static("Fraw",[](const int b, const int l, const int dev){
    return SO3part<float>::Fraw(b,l,dev);}, 
    py::arg("b"), py::arg("l"), py::arg("device")=0)

  .def_static("Fzero",[](const int b, const int l, const int dev){
      return SO3part<float>::Fzero(b,l,dev);}, 
    py::arg("b"), py::arg("l"), py::arg("device")=0)

  .def_static("Fgaussian",[](const int b, const int l, const int dev){
      return SO3part<float>::Fgaussian(b,l,dev);}, 
    py::arg("b"), py::arg("l"), py::arg("device")=0)

  .def_static("zeros_like",[](const SO3part<float>& x){return SO3part<float>(SO3part<float>::zeros_like(x));})
*/

//py::class_<SO3partView<float> >(m,"SO3partView");
//py::class_<TensorVirtual<complex<float>, SO3partView<float> > >(m,"VirtualSO3part");


