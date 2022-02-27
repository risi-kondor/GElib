// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partC
#define _SO3partC

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


  // An SO3partC is a  b x (2l+1) x n   dimensional complex tensor.


  class SO3partC: public cnine::CtensorB{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;

    
    using CtensorB::CtensorB;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3partC(const Gdims& _adims, const int l, const int n, const int _dev=0):
      CtensorB(Gdims(_adims, Gdims({2*l+1,n})),_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partC(const Gdims& adims, const int l, const int n, const FILLTYPE& dummy, const int _dev=0):
      CtensorB(Gdims(adims,Gidms({2*l+1,n})),dummy,_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3partC zero(const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partC(_adims,l,n,cnine::fill_zero(),_dev);
    }

    static SO3partC gaussian(const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partC(_adims,l,n,cnine::fill_gaussian(),_dev);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partC(const CtensorB& x):
      CtensorB(x){
      assert(dims.back(1)%2==1);
    }
      
    SO3partC(CtensorB&& x):
      CtensorB(std::move(x)){
      assert(dims.back(1)%2==1);
    }
      

  public: // ---- Access -------------------------------------------------------------------------------------


    Gdims get_adims() const{
      return dims.chunk(0,dims.size()-2);
    }
    
    int getb() const{
      int t=0;
      for(int i=0; i<dims.size()-2; i++)
	t*=dims(i);
      return t;
    }

    int getl() const{
      return (dims.back(1)-1)/2;
    }

    int getn() const{
      return dims.last();
    }

    Gdims get_cdims(){
      return Gdims({getb(),getl(),getn()});
    }

    Gstrides get_cstrides(){
      return strides.chunk(strides.size()-3);
    }

        
  public: // ---- Access views --------------------------------------------------------------------------------


    Ctensor3_view view3D() const{
      return Ctensor3_view(arr,get_cdims(),get_cstrides(),coffs);
    }

    SO3part3_view Pview3D() const{
      if(dev==0) return SO3part3_view(arr,get_cdims(),get_cstrides(),coffs);
      else return SO3part3_view(arrg,get_cdims(),get_cstrides(),coffs,dev);
    }

    //operator SO3part3_view() const{
    //if(dev==0) return SO3part3_view(arr,get_cdims(),get_cstrides(),coffs);
    //else return SO3part3_view(arrg,get_cdims(),get_cstrides(),coffs,dev);
    //}

    SO3Fpart3_view Fview3D() const{
      if(dev==0) return SO3Fpart3_view(arr,get_cdims(),get_cstrides(),coffs);
      else return SO3Fpart3_view(arrg,get_cdims(),get_cstrides(),coffs,dev);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------

k
    //SO3partC operator-(const SO3partC& y) const{
    //return (*this)-y;
    //}


  public: // ---- Rotations ----------------------------------------------------------------------------------


    SO3partC rotate(const SO3element& r){
      CtensorB D(WignerMatrix<float>(getl(),r),dev);
      SO3partC R=SO3partC::zero(get_adims(),getl(),getn(),dev);

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

    
    SO3partC CGproduct(const SO3partC& y, const int l) const{
      assert(l>=abs(getl()-y.getl()) && l<=getl()+y.getl());
      SO3partC R=SO3partC::zero(get_adims(),l,getn()*y.getn(),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }


    void add_CGproduct(const SO3partC& x, const SO3partC& y, const int _offs=0){
      auto v=this->Pview3D();
      SO3part_addCGproductFn()(v,x.Pview3D(),y.Pview3D(),_offs);
    }

    void add_CGproduct_back0(const SO3partC& g, const SO3partC& y, const int _offs=0){
      auto v=this->Pview3D();
      SO3part_addCGproduct_back0Fn()(v,g.Pview3D(),y.Pview3D(),_offs);
    }

    void add_CGproduct_back1(const SO3partC& g, const SO3partC& x, const int _offs=0){
      auto v=this->Pview3D();
      SO3part_addCGproduct_back1Fn()(v,g.Pview3D(),x.Pview3D(),_offs);
    }

  };

}

#endif 
