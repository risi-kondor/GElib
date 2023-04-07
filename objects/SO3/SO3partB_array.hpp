// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor 
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partB_array
#define _SO3partB_array

#include "CtensorArrayB.hpp"
#include "SO3part3_view.hpp"
#include "SO3partB.hpp"

#include "SO3part_addSpharmFn.hpp"
#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"
#include "SO3part_addCGsquareFn.hpp"
#include "SO3part_addFproduct_Fn.hpp"
#include "SO3part_addFproduct_back0Fn.hpp"
#include "SO3part_addFproduct_back1Fn.hpp"

#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{


  // An SO3partB_array is an (a1 x ... x ak) x N x b x (2l+1) x n   dimensional complex tensor.


  class SO3partB_array: public cnine::CtensorArrayB{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;
    typedef cnine::Gstrides Gstrides;
    typedef cnine::Ctensor3_view Ctensor3_view;

    using CtensorArrayB::CtensorArrayB;

    
  public: // ---- Constructors -------------------------------------------------------------------------------


    //SO3partB_array(const int N, const int b, const int l, const int n, const int _dev=0):
    //CtensorArrayB({N},{b,2*l+1,n},_dev){}

    //template<typename FILLTYPE, typename = typename 
    //     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //SO3partB_array(const int N, const int b, const int l, const int n, const FILLTYPE& dummy, const int _dev=0):
    //CtensorArrayB({N},{b,2*l+1,n},dummy,_dev){}

    SO3partB_array(const Gdims& _adims, const int l, const int n, const int _dev=0):
      CtensorArrayB(_adims,{2*l+1,n},_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partB_array(const Gdims& _adims, const int l, const int n, const FILLTYPE& dummy, const int _dev=0):
      CtensorArrayB(_adims,{2*l+1,n},dummy,_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partB_array(const int b, const Gdims& _adims, const int l, const int n, const FILLTYPE& dummy, const int _dev=0):
      CtensorArrayB(_adims.prepend(b),{2*l+1,n},dummy,_dev){
      batched=true;
    }

    

  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3partB_array zero(const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partB_array(_adims,l,n,cnine::fill_zero(),_dev);
    }

    static SO3partB_array ones(const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partB_array(_adims,l,n,cnine::fill_zero(),_dev);
    }

    static SO3partB_array gaussian(const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partB_array(_adims,l,n,cnine::fill_gaussian(),_dev);
    }


    static SO3partB_array zero(const int b, const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partB_array(b,_adims,l,n,cnine::fill_zero(),_dev);
    }

    static SO3partB_array ones(const int b, const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partB_array(b,_adims,l,n,cnine::fill_zero(),_dev);
    }

    static SO3partB_array gaussian(const int b, const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partB_array(b,_adims,l,n,cnine::fill_gaussian(),_dev);
    }

    
    static SO3partB_array zeros_like(const SO3partB_array& x){
      return SO3partB_array(x.getb(),x.get_adims(),x.getl(),x.getn(),cnine::fill_zero(),x.dev);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    SO3partB_array(const SO3partB_array& x):
      CtensorArrayB(x){
      //CtensorArrayB(x,-2){
      GELIB_COPY_WARNING();
    }
      
    SO3partB_array(SO3partB_array&& x):
      CtensorArrayB(std::move(x)){
      //CtensorArrayB(std::move(x),-2){
      GELIB_MOVE_WARNING();
    }

      
  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partB_array(const CtensorB& x):
      CtensorArrayB(x,-2){
      GELIB_CONVERT_WARNING(x);
    }
      
    SO3partB_array(CtensorB&& x):
      CtensorArrayB(std::move(x),-2){
      GELIB_MCONVERT_WARNING(x);
    }

    SO3partB_array(const CtensorArrayB& x):
      CtensorArrayB(x){
      GELIB_CONVERT_WARNING(x);
    }
      
    SO3partB_array(CtensorArrayB&& x):
      CtensorArrayB(std::move(x)){
      GELIB_MCONVERT_WARNING(x);
    }

      
  public: // ---- ATen --------------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


    int getN() const{
      return memsize/strides.back(2);
    }
    
    int getb() const{
      return dims(0);
    }

    int getl() const{
      return (dims.back(1)-1)/2;
    }

    int getn() const{
      return dims.back(0);
    }

    Gdims get_adims() const{
      return dims.chunk(1,ak-1);
    }

    //Gstrides get_cstrides(){
    //return strides.chunk(strides.size()-3);
    //}

        
  public: // ---- Access views --------------------------------------------------------------------------------

    /*
    Ctensor3_view view3D() const{
      return Ctensor3_view(arr,Gdims({dims(0)*dims(1),dims(2),dims(3)}),Gstrides({strides[1],strides[2],strides[3]}),coffs);
    }
    */

    /*
    SO3part3_view Pview3D() const{
      if(dev==0) return SO3part3_view(arr,Gdims({GetN(),getl(),getn()}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs);
      else return SO3part3_view(arrg,Gdims({getN(),getl(),getn()}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs,dev);
    }
    */

    SO3partB fused_view(){
      return SO3partB(view_fusing_first(ak));
    }

    const SO3partB fused_view() const{
      return SO3partB(const_cast<SO3partB_array*>(this)->view_fusing_first(ak));
    }

    SO3part3_view part3_view() const{
      if(dev==0) return SO3part3_view(arr,Gdims({getN(),dims.back(1),dims.back(0)}),
	Gstrides({strides.back(2),strides.back(1),strides.back(0)}),coffs);
      else return SO3part3_view(arrg,Gdims({getN(),dims.back(1),dims.back(0)}),
	Gstrides({strides.back(2),strides.back(1),strides.back(0)}),coffs,dev);
    }

    /*
    SO3part3_view cell_view(const int i) const{
      return SO3part3_view(arr+strides[0]*i,{dims(1),dims(2),dims(3)},{strides[1],strides[2],strides[3]},coffs,dev);
    }
    */

    //operator SO3part3_view() const{
    //if(dev==0) return SO3part3_view(arr,get_cdims(),get_cstrides(),coffs);
    //else return SO3part3_view(arrg,get_cdims(),get_cstrides(),coffs,dev);
    //}

    /*
    SO3Fpart3_view Fview3D() const{
      if(dev==0) return SO3Fpart3_view(arr,Gdims({dims(0)*dims(1),dims(2),dims(3)}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs);
      else return SO3Fpart3_view(arrg,Gdims({dims(0)*dims(1),dims(2),dims(3)}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs,dev);
    }
    */

  public: // ---- Operations ---------------------------------------------------------------------------------


    //SO3partB_array operator-(const SO3partB_array& y) const{
    //return (*this)-y;
    //}


  public: // ---- Rotations ----------------------------------------------------------------------------------


    SO3partB_array rotate(const SO3element& r) const{
      CtensorB D(WignerMatrix<float>(getl(),r),dev);
      SO3partB_array R=SO3partB_array::zeros_like(*this);

      auto dv=D.view2D();
      auto xv=fused_view().view3();
      auto rv=R.fused_view().view3();
      
      int B=rv.n0;
      for(int b=0; b<B; b++){
	auto v=rv.slice0(b);
	cnine::Ctensor_add_mprod_AA()(v,dv,xv.slice0(b));
      }

      return R;
    }



  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add_gather(const SO3partB_array& x, const cnine::Rmask1& mask){
      if(!batched) CtensorArrayB::add_gather(x,mask);
      else{
	CNINE_ASSRT(dims[0]==x.dims[0]);
	for(int i=0; i<dims[0]; i++){
	  //cout<<"ddd"<<x.viewd()<<endl;
	  cnine::Aggregator(viewd().slice0(i),x.viewd().slice0(i),mask);
	}
      }
    }

    /*
    SO3partB_array mprod(const CtensorB& y){
      assert(y.ndims()==2);
      assert(y.dims(0)==getn());
      SO3partB_array R=SO3partB::zero(get_adims(),getl(),y.dims(1),dev);
      R.add_mprod(*this,y);
      return R;
    }

    void add_mprod(const SO3partB& x, const CtensorB& w){
      view3().fuse01().add_matmul(x.view3().fuse01(),w.view2());
    }

    void add_mprod_back0(const SO3partB& rg, const CtensorB& w){
      view3().fuse01().add_matmul_AH(rg.view3().fuse01(),w.view2());
    }

    void add_mprod_back1_into(CtensorB& yg, const SO3partB& x) const{
      yg.view2().add_matmul_HA(x.view3().fuse01(),view3().fuse01());
    }
    */
    
    /*
    add_convolve(const SO3partB_array& x, const CSRmatrix<float>& M){
      GELIB_ASSRT(batched==x.batched);
      GELIB_ASSRT(ak==x.ak);

      if(ak-batched==2){
	if(batched){
	  Rtensor5_view rv(arr,);
	}
      }

    }
    */


  public: // ---- Spherical harmonics -----------------------------------------------------------------------


    static SO3partB_array spharm(const int l, const cnine::RtensorA& x, const int _dev=0, const bool _batched=0){
      if(_batched){
	assert(x.ndims()>=4);
	SO3partB_array R(x.dims[0],x.dims.chunk(1,x.ndims()-3),l,x.dims(-1),cnine::fill_zero());
	R.add_spharm(x);
	//SO3part_addSpharmFn()(R.part3_view(),x.view());
	if(_dev>0) return SO3partB_array(R,_dev);
	return R;
      }
      assert(x.ndims()>=3);
      SO3partB_array R(x.dims.chunk(0,x.ndims()-2),l,x.dims(-1),cnine::fill_zero());
      R.add_spharm(x);
      //SO3part_addSpharmFn()(R.part3_view(),x.view());
      if(_dev>0) return SO3partB_array(R,_dev);
      return R;
    }


    void add_spharm(const cnine::RtensorA& x){
      auto v=part3_view();
      SO3part_addSpharmFn()(v,x.viewx().fuse_all_but_last_two());
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    SO3partB_array CGproduct(const SO3partB_array& y, const int l) const{
      assert(l>=abs(getl()-y.getl()) && l<=getl()+y.getl());
      SO3partB_array R=SO3partB_array::zeros_like(*this);
      //SO3partB_array R=SO3partB_array::zero(get_adims(),l,getn()*y.getn(),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }


    void add_CGproduct(const SO3partB_array& x, const SO3partB_array& y, const int _offs=0){
      auto adims=get_adims();
      auto xadims=x.get_adims();
      auto yadims=y.get_adims();

      if(adims==xadims && adims==yadims){
	auto v=this->part3_view();
	SO3part_addCGproductFn()(v,x.part3_view(),y.part3_view(),_offs);
	return; 
      }

      /*
      if(adims.size()==2){
	if(xadims.size()==1 && yadims.size()==2 && xadims[0]==yadims[0]){
	  CNINE_ASSRT(adims[0]==yadims[0] && adims[1]==yadims[1]);
	  for(int i=0; i<yadims[1]; i++){
	    SO3part_addCGproductFn()(this->viewx().slice1(i),x.part3_view(),y.part3_view(),_offs);
	  }
	}
      }
      */
    }

    void add_CGproduct_back0(const SO3partB_array& g, const SO3partB_array& y, const int _offs=0){
      auto v=this->part3_view();
      SO3part_addCGproduct_back0Fn()(v,g.part3_view(),y.part3_view(),_offs);
    }

    void add_CGproduct_back1(const SO3partB_array& g, const SO3partB_array& x, const int _offs=0){
      auto v=this->part3_view();
      SO3part_addCGproduct_back1Fn()(v,g.part3_view(),x.part3_view(),_offs);
    }


    // ---- BlockedCGproduct ---------------------------------------------------------------------------------


    SO3partB_array BlockedCGproduct(const SO3partB_array& y, const int bsize, const int l) const{
      assert(l>=abs(getl()-y.getl()) && l<=getl()+y.getl());
      assert(getn()==y.getn());
      SO3partB_array R=SO3partB_array::zeros_like(*this);
      R.add_BlockedCGproduct(*this,y,bsize);
      return R;
    }

    void add_BlockedCGproduct(const SO3partB_array& x, const SO3partB_array& y, const int bsize, const int _offs=0){
      SO3part_addBlockedCGproductFn()(part3_view(),x.part3_view(),y.part3_view(),bsize,_offs);
    }

    void add_BlockedCGproduct_back0(const SO3partB_array& g, const SO3partB_array& y, const int bsize, const int _offs=0){
      SO3part_addBlockedCGproduct_back0Fn()(part3_view(),g.part3_view(),y.part3_view(),bsize,_offs);
    }

    void add_BlockedCGproduct_back1(const SO3partB_array& g, const SO3partB_array& x, const int bsize, const int _offs=0){
      SO3part_addBlockedCGproduct_back1Fn()(part3_view(),g.part3_view(),x.part3_view(),bsize,_offs);
    }


    // ---- DiagCGproduct 


    SO3partB_array DiagCGproduct(const SO3partB_array& y, const int l) const{
      return BlockedCGproduct(y,1,l);
    }

    void add_DiagCGproduct(const SO3partB_array& x, const SO3partB_array& y, const int _offs=0){
      add_BlockedCGproduct(x,y,1,_offs);
    }

    void add_DiagCGproduct_back0(const SO3partB_array& g, const SO3partB_array& y, const int _offs=0){
      add_BlockedCGproduct_back0(g,y,1,_offs);
    }

    void add_DiagCGproduct_back1(const SO3partB_array& g, const SO3partB_array& x, const int _offs=0){
      add_BlockedCGproduct_back1(g,x,1,_offs);
    }




    SO3partB_array CGsquare(const int l) const{
      assert(l>=0 && l<=2*getl());
      int parity=(2*getl()-l)%2;
      SO3partB_array R=SO3partB_array::zeros_like(*this);
      //SO3partB_array R=SO3partB_array::zero(get_adims(),l,getn()*(getn()+1-2*parity)/2,get_dev());
      R.add_CGsquare(*this);
      return R;
    }

    void add_CGsquare(const SO3partB_array& x, const int _offs=0){
      SO3part_addCGsquareFn()(part3_view(),x.part3_view(),_offs);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    /*
    string str(const string indent="") const{
      ostringstream oss;
      int N=getN();
      for(int i=0; i<N; i++){
	oss<<indent<<"Cell "<<i<<":"<<endl;
	oss<<cell_view(i).str(indent+"  ")<<endl;
      }
      return oss.str();
    }
    */
    
    string classname() const{
      return "GElib::SO3partB_array";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3partB_array of type ("+to_string(getb())+","+get_adims().str()+","+to_string(getl())+","+to_string(getn())+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3partB_array& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
