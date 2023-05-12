
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

  .def_static("zero",[](const int b, const Gdims& adims, const int l, const int n, const int dev){
      return SO3partB_array::zero(b,adims,l,n,dev);}, 
    py::arg("b"), py::arg("adims"), py::arg("l"), py::arg("n"), py::arg("device")=0)
  .def_static("zero",[](const int b, const vector<int>& av, const int l, int n, const int dev){
      return SO3partB_array::zero(b,Gdims(av),l,n,dev);},
    py::arg("b"), py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

  .def_static("gaussian",[](const int b, const Gdims& adims, const int l, const int n, const int dev){
      return SO3partB_array::gaussian(b,adims,l,n,dev);}, 
    py::arg("b"), py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)
  .def_static("gaussian",[](const int b, const vector<int>& av, const int l, const int n, const int dev){
      return SO3partB_array::gaussian(b,Gdims(av),l,n,dev);},
    py::arg("b"), py::arg("adims"),py::arg("l"),py::arg("n"),py::arg("device")=0)

//.def_static("view",[](at::Tensor& x){return SO3partB_array(cnine::CtensorB::view(x));})
  .def_static("view",[](at::Tensor& x){return SO3partB_array(SO3partB_array::view(x,-2,true));})
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
  .def("add_gather",[](SO3partB_array& r, const SO3partB_array& x, const cnine::Rmask1& mask){
      r.add_gather(x,mask);})

  .def("add_spharm",[](SO3partB_array& obj, at::Tensor& _X){
      RtensorA X=RtensorA::view(_X);
      obj.add_spharm(X);})

  .def("addCGproduct",&SO3partB_array::add_CGproduct,py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("addCGproduct_back0",&SO3partB_array::add_CGproduct_back0,py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("addCGproduct_back1",&SO3partB_array::add_CGproduct_back1,py::arg("g"),py::arg("x"),py::arg("offs")=0)

  .def("addDiagCGproduct",&SO3partB_array::add_DiagCGproduct,py::arg("x"),py::arg("y"),py::arg("offs")=0)
  .def("addDiagCGproduct_back0",&SO3partB_array::add_DiagCGproduct_back0,py::arg("g"),py::arg("y"),py::arg("offs")=0)
  .def("addDiagCGproduct_back1",&SO3partB_array::add_DiagCGproduct_back1,py::arg("g"),py::arg("x"),py::arg("offs")=0)

  .def("add_conterpolate2d",[](SO3partB_array& r, const SO3partB_array& x, const RtensorObj& M){
      
      GELIB_ASSERT(M.ndims()>=3,"2D conterpolation tensor must have two spatial dimensions and at least one output dimension");
      GELIB_ASSERT(M.dims(-2)%2==1&&M.dims(-1)%2==1,"Conterpolation tensor's spatial dimensions must be odd");

      GELIB_ASSRT(x.ndims()==5);
      GELIB_ASSRT(r.ndims()-5==M.ndims()-2);

      GELIB_ASSRT(r.dims[0]==x.dims[0]);
      GELIB_ASSRT(r.dims[1]==x.dims[1]);
      GELIB_ASSRT(r.dims[2]==x.dims[2]);
      for(int i=0; i<M.ndims()-2; i++)
	GELIB_ASSERT(r.dims[3+i]==M.dims[i],"Mismatch between output dimensions and dimensions of conterpolation matrix");
      GELIB_ASSRT(r.dims(-2)==x.dims(-2));
      GELIB_ASSRT(r.dims(-1)==x.dims(-1));

      int aoutd=1; for(int i=0; i<M.ndims()-2; i++) aoutd*=M.dims[i];
      
      Ctensor5_view xv(x.get_arr(),x.get_arr()+x.coffs,
	x.dims[0],x.dims[1],x.dims[2],1,x.dims[3]*x.dims[4],
	x.strides[0],x.strides[1],x.strides[2],x.strides[2],x.strides[4],x.dev);

      Ctensor5_view rv(r.get_arr(),r.get_arr()+r.coffs,
	r.dims[0],r.dims[1],r.dims[2],aoutd,r.dims(-2)*r.dims(-1), 
 	r.strides[0],r.strides[1],r.strides[2],r.strides(-3),r.strides(-1),r.dev);
      
      Rtensor4_view Mv(M.get_arr(),aoutd,M.dims(-2),M.dims(-1),1,M.strides(-3),M.strides(-2),M.strides(-1),M.strides(-1),M.dev);

      CtensorConvolve2d()(rv,xv,Mv);
    })

  .def("add_conterpolate2d_back",[](SO3partB_array& x, const SO3partB_array& r, const RtensorObj& M){
      GELIB_ASSERT(M.ndims()>=3,"2D conterpolation tensor must have two spatial dimensions and at least one output dimension");
      GELIB_ASSERT(M.dims(-2)%2==1&&M.dims(-1)%2==1,"Conterpolation tensor's spatial dimensions must be odd");

      GELIB_ASSRT(x.ndims()==5);
      GELIB_ASSRT(r.ndims()-5==M.ndims()-2);

      GELIB_ASSRT(r.dims[0]==x.dims[0]);
      GELIB_ASSRT(r.dims[1]==x.dims[1]);
      GELIB_ASSRT(r.dims[2]==x.dims[2]);
      for(int i=0; i<M.ndims()-2; i++)
	GELIB_ASSERT(r.dims[3+i]==M.dims[i],"Mismatch between output dimensions and dimensions of conterpolation matrix");
      GELIB_ASSRT(r.dims(-2)==x.dims(-2));
      GELIB_ASSRT(r.dims(-1)==x.dims(-1));

      int aoutd=1; for(int i=0; i<M.ndims()-2; i++) aoutd*=M.dims[i];

      Ctensor5_view xv(x.get_arr(),x.get_arr()+x.coffs,
	x.dims[0],x.dims[1],x.dims[2],1,x.dims[3]*x.dims[4],
	x.strides[0],x.strides[1],x.strides[2],x.strides[2],x.strides[4],x.dev);

      Ctensor5_view rv(r.get_arr(),r.get_arr()+r.coffs,
	r.dims[0],r.dims[1],r.dims[2],aoutd,r.dims(-2)*r.dims(-1),
	r.strides[0],r.strides[1],r.strides[2],r.strides(-3),r.strides(-1),r.dev);
      
      Rtensor4_view Mv(M.get_arr(), 1,M.dims(-2),M.dims(-1),aoutd,
	M.strides(-1),M.strides(-2),M.strides(-1),M.strides(-3),M.dev);

      CtensorConvolve2d()(xv,rv,Mv);      
    })

  .def("add_conterpolate3d",[](SO3partB_array& r, const SO3partB_array& x, const RtensorObj& M){

      GELIB_ASSERT(M.ndims()>=4,"3D conterpolation tensor must have three spatial dimensions and at least one output dimension");
      GELIB_ASSERT(M.dims(-3)%2==1&&M.dims(-2)%2==1&&M.dims(-1)%2==1,"Conterpolation tensor's spatial dimensions must be odd");

      GELIB_ASSRT(x.ndims()==6);
      GELIB_ASSRT(r.ndims()-6==M.ndims()-3);

      GELIB_ASSRT(r.dims[0]==x.dims[0]);
      GELIB_ASSRT(r.dims[1]==x.dims[1]);
      GELIB_ASSRT(r.dims[2]==x.dims[2]);
      GELIB_ASSRT(r.dims[3]==x.dims[3]);
      for(int i=0; i<M.ndims()-3; i++)
	GELIB_ASSERT(r.dims[4+i]==M.dims[i],"Mismatch between output dimensions and dimensions of conterpolation matrix");
      GELIB_ASSRT(r.dims(-2)==x.dims(-2));
      GELIB_ASSRT(r.dims(-1)==x.dims(-1));

      int aoutd=1; for(int i=0; i<M.ndims()-3; i++) aoutd*=M.dims[i];
      
      Ctensor6_view xv(x.true_arr(),x.get_arr()+x.coffs,
	x.dims[0],x.dims[1],x.dims[2],x.dims[3],1,x.dims[4]*x.dims[5],
	x.strides[0],x.strides[1],x.strides[2],x.strides[3],x.strides[3],x.strides[5],x.dev);

      Ctensor6_view rv(r.true_arr(),r.get_arr()+r.coffs,
	r.dims[0],r.dims[1],r.dims[2],r.dims[3],aoutd,r.dims(-2)*r.dims(-1),
	r.strides[0],r.strides[1],r.strides[2],r.strides[3],r.strides(-3),r.strides(-1),r.dev);
      
      Rtensor5_view Mv(M.get_arr(),aoutd,M.dims(-3),M.dims(-2),M.dims(-1),1,
	M.strides(-4),M.strides(-3),M.strides(-2),M.strides(-1),M.strides(-1),M.dev);

      CtensorConvolve3d()(rv,xv,Mv);
    })

  .def("add_conterpolate3d_back",[](SO3partB_array& x, const SO3partB_array& r, const RtensorObj& M){
      GELIB_ASSERT(M.ndims()>=4,"3D conterpolation tensor must have three spatial dimensions and at least one output dimension");
      GELIB_ASSERT(M.dims(-3)%2==1&&M.dims(-2)%2==1&&M.dims(-1)%2==1,"Conterpolation tensor's spatial dimensions must be odd");

      GELIB_ASSRT(x.ndims()==6);
      GELIB_ASSRT(r.ndims()-6==M.ndims()-3);

      GELIB_ASSRT(r.dims[0]==x.dims[0]);
      GELIB_ASSRT(r.dims[1]==x.dims[1]);
      GELIB_ASSRT(r.dims[2]==x.dims[2]);
      GELIB_ASSRT(r.dims[3]==x.dims[3]);
      for(int i=0; i<M.ndims()-3; i++)
	GELIB_ASSERT(r.dims[4+i]==M.dims[i],"Mismatch between output dimensions and dimensions of conterpolation matrix");
      GELIB_ASSRT(r.dims(-2)==x.dims(-2));
      GELIB_ASSRT(r.dims(-1)==x.dims(-1));

      int aoutd=1; for(int i=0; i<M.ndims()-3; i++) aoutd*=M.dims[i];

      Ctensor6_view xv(x.true_arr(),x.true_arr()+x.coffs,
	x.dims[0],x.dims[1],x.dims[2],x.dims[3],1,x.dims[4]*x.dims[5],
	x.strides[0],x.strides[1],x.strides[2],x.strides[3],x.strides[3],x.strides[5],x.dev);

      Ctensor6_view rv(r.true_arr(),r.true_arr()+r.coffs,
	r.dims[0],r.dims[1],r.dims[2],r.dims[3],aoutd,r.dims(-2)*r.dims(-1),
	r.strides[0],r.strides[1],r.strides[2],r.strides[3],r.strides(-3),r.strides(-1),r.dev);
      
      Rtensor5_view Mv(M.get_arr(),1,M.dims(-3),M.dims(-2),M.dims(-1),aoutd,
	M.strides(-1),M.strides(-3),M.strides(-2),M.strides(-1),M.strides(-4),M.dev);

      CtensorConvolve3d()(xv,rv,Mv);      
    })

  .def("device",&SO3partB_array::get_device)
  .def("to",&SO3partB_array::to_device)
  .def("to_device",&SO3partB_array::to_device)
  .def("move_to",[](SO3partB_array& x, const int _dev){x.move_to_device(_dev);})

  .def("str",&SO3partB_array::str,py::arg("indent")="")
  .def("__str__",&SO3partB_array::str,py::arg("indent")="")
  .def("__repr__",&SO3partB_array::repr,py::arg("indent")="");


// ---- Stand-alone functions --------------------------------------------------------------------------------



