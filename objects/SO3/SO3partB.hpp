// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Gpart
#define _SO3Gpart

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


  // An SO3partB is a  b x (2l+1) x n   dimensional complex tensor.


  class SO3partB: public cnine::CtensorB{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;

    
    using CtensorB::CtensorB;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3partB(const int b, const int l, const int n, const int _dev=0):
      CtensorB(Gdims(b,2*l+1,n),_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partB(const int b, const int l, const int n, const FILLTYPE& dummy, const int _dev=0):
      CtensorB(Gdims(b,2*l+1,n),dummy,_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3partB zero(const int b, const int l, const int n,  const int _dev=0){
      return SO3partB(b,l,n,cnine::fill_zero(),_dev);
    }

    static SO3partB gaussian(const int b, const int l, const int n,  const int _dev=0){
      return SO3partB(b,l,n,cnine::fill_gaussian(),_dev);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partB(const CtensorB& x):
      CtensorB(x){
      assert(dims.size()==3);
      assert(dims(1)%2==1);
    }
      
    SO3partB(CtensorB&& x):
      CtensorB(std::move(x)){
      assert(dims.size()==3);
      assert(dims(1)%2==1);
    }
      

  public: // ---- Access -------------------------------------------------------------------------------------

    int getb() const{
      return dims(0);
    }
    
    int getl() const{
      return (dims(1)-1)/2;
    }

    int getn() const{
      return dims(2);
    }

        
  public: // ---- Access views --------------------------------------------------------------------------------


    SO3part3_view view() const{
      if(dev==0) return SO3part3_view(arr,dims,strides,coffs);
      else return SO3part3_view(arrg,dims,strides,coffs,dev);
    }

    operator SO3part3_view() const{
      if(dev==0) return SO3part3_view(arr,dims,strides,coffs);
      else return SO3part3_view(arrg,dims,strides,coffs,dev);
    }


    SO3Fpart3_view Fview() const{
      return SO3Fpart3_view(arr,dims,strides,coffs);
    }


  public: // ---- Rotations ----------------------------------------------------------------------------------


    SO3partB rotate(const SO3element& r){
      CtensorB D(WignerMatrix<float>(getl(),r),dev);
      SO3partB R=SO3partB::zero(getb(),getl(),getn(),dev);

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

    
    SO3partB CGproduct(const SO3partB& y, const int l) const{
      assert(l>=abs(getl()-y.getl()) && l<=getl()+y.getl());
      SO3partB R=SO3partB::zero(getb(),l,getn()*y.getn(),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }


    void add_CGproduct(const SO3partB& x, const SO3partB& y, const int _offs=0){
      auto v=this->view();
      SO3part_addCGproductFn()(v,x,y,_offs);
    }

    void add_CGproduct_back0(const SO3partB& g, const SO3partB& y, const int _offs=0){
      auto v=this->view();
      SO3part_addCGproduct_back0Fn()(v,g,y,_offs);
    }

    void add_CGproduct_back1(const SO3partB& g, const SO3partB& x, const int _offs=0){
      auto v=this->view();
      SO3part_addCGproduct_back1Fn()(v,g,x,_offs);
    }


  };

}

#endif 
