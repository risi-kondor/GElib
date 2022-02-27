// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partD
#define _SO3partD

#include "CtensorB.hpp"
#include "SO3part3_view.hpp"
#include "SO3Fpart3_view.hpp"
#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"
//#include "SO3_CGbank.hpp"
//#include "SO3_SPHgen.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{


  // An SO3partD is an  N x b x (2l+1) x n   dimensional complex tensor.


  class SO3partD: public cnine::CtensorB{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;
    typedef cnine::Gstrides Gstrides;
    typedef cnine::Ctensor3_view Ctensor3_view;

    
    using CtensorB::CtensorB;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3partD(const int N, const int b, const int l, const int n, const int _dev=0):
      CtensorB({N,b,2*l+1,n},_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partD(const int N, const int b, const int l, const int n, const FILLTYPE& dummy, const int _dev=0):
      CtensorB({N,b,2*l+1,n},dummy,_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3partD zero(const int N, const int b, const int l, const int n,  const int _dev=0){
      return SO3partD(N,b,l,n,cnine::fill_zero(),_dev);
    }

    static SO3partD gaussian(const int N, const int b, const int l, const int n,  const int _dev=0){
      return SO3partD(N,b,l,n,cnine::fill_gaussian(),_dev);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partD(const CtensorB& x):
      CtensorB(x){
      assert(dims.size()==4);
      assert(dims(2)%2==1);
    }
      
    SO3partD(CtensorB&& x):
      CtensorB(std::move(x)){
      assert(dims.size()==4);
      assert(dims(2)%2==1);
    }
      

  public: // ---- Access -------------------------------------------------------------------------------------


    int getN() const{
      return dims(0);
    }
    
    int getb() const{
      return dims(1);
    }

    int getl() const{
      return (dims(2)-1)/2;
    }

    int getn() const{
      return dims(3);
    }

    //Gdims get_cdims(){
    //return Gdims({getb(),getl(),getn()});
    //}

    //Gstrides get_cstrides(){
    //return strides.chunk(strides.size()-3);
    //}

        
  public: // ---- Access views --------------------------------------------------------------------------------


    Ctensor3_view view3D() const{
      return Ctensor3_view(arr,Gdims({dims(0)*dims(1),dims(2),dims(3)}),Gstrides({strides[1],strides[2],strides[3]}),coffs);
    }

    SO3part3_view Pview3D() const{
      if(dev==0) return SO3part3_view(arr,Gdims({dims(0)*dims(1),dims(2),dims(3)}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs);
      else return SO3part3_view(arrg,Gdims({dims(0)*dims(1),dims(2),dims(3)}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs,dev);
    }

    SO3part3_view cell_view(const int i) const{
      return SO3part3_view(arr+strides[0]*i,{dims(1),dims(2),dims(3)},{strides[1],strides[2],strides[3]},coffs,dev);
    }

    //operator SO3part3_view() const{
    //if(dev==0) return SO3part3_view(arr,get_cdims(),get_cstrides(),coffs);
    //else return SO3part3_view(arrg,get_cdims(),get_cstrides(),coffs,dev);
    //}

    SO3Fpart3_view Fview3D() const{
      if(dev==0) return SO3Fpart3_view(arr,Gdims({dims(0)*dims(1),dims(2),dims(3)}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs);
      else return SO3Fpart3_view(arrg,Gdims({dims(0)*dims(1),dims(2),dims(3)}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs,dev);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    //SO3partD operator-(const SO3partD& y) const{
    //return (*this)-y;
    //}


  public: // ---- Rotations ----------------------------------------------------------------------------------


    SO3partD rotate(const SO3element& r){
      CtensorB D(WignerMatrix<float>(getl(),r),dev);
      SO3partD R=SO3partD::zero(getN(),getb(),getl(),getn(),dev);

      int B=getb();
      auto dv=D.view2D();
      auto xv=this->view3D();
      auto rv=R.view3D();
      
      for(int b=0; b<B; b++){
	auto v=rv.slice0(b);
	cnine::Ctensor_add_mprod_AA()(v,dv,xv.slice0(b));
      }

      return R;
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    SO3partD CGproduct(const SO3partD& y, const int l) const{
      assert(l>=abs(getl()-y.getl()) && l<=getl()+y.getl());
      SO3partD R=SO3partD::zero(getN(),getb(),l,getn()*y.getn(),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }


    void add_CGproduct(const SO3partD& x, const SO3partD& y, const int _offs=0){
      auto v=this->Pview3D();
      SO3part_addCGproductFn()(v,x.Pview3D(),y.Pview3D(),_offs);
    }

    void add_CGproduct_back0(const SO3partD& g, const SO3partD& y, const int _offs=0){
      auto v=this->Pview3D();
      SO3part_addCGproduct_back0Fn()(v,g.Pview3D(),y.Pview3D(),_offs);
    }

    void add_CGproduct_back1(const SO3partD& g, const SO3partD& x, const int _offs=0){
      auto v=this->Pview3D();
      SO3part_addCGproduct_back1Fn()(v,g.Pview3D(),x.Pview3D(),_offs);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      int N=getN();
      for(int i=0; i<N; i++){
	oss<<indent<<"Cell "<<i<<":"<<endl;
	oss<<cell_view(i).str(indent+"  ")<<endl;
      }
      return oss.str();
    }

    
  };

}

#endif 
