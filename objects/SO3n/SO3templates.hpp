
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3templates
#define _GElibSO3templates

#include "GElib_base.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"

#include "SO3part_addBlockedCGproductFn.hpp"
#include "SO3part_addBlockedCGproduct_back0Fn.hpp"
#include "SO3part_addBlockedCGproduct_back1Fn.hpp"


namespace GElib{

  class SO3part_t{};
  class SO3vec_t{};


  //template<typename TYPE0, typename TYPE1, typename = typename 
  //std::enable_if<std::is_base_of<SO3part_t, TYPE1>::value, TYPE1>::type>
  template<typename TYPE0, typename TYPE1>
  void add_CGproduct(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addCGproductFn()(r,x,y,offs);
  }

  template<typename TYPE0, typename TYPE1>
  void add_CGproduct_back0(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addCGproduct_back0Fn()(r,x,y,offs);
  }

  template<typename TYPE0, typename TYPE1>
  void add_CGproduct_back1(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addCGproduct_back1Fn()(r,x,y,offs);
  }


  template<typename TYPE0, typename TYPE1>
  void add_DiagCGproduct(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addBlockedCGproductFn()(r,x,y,1,offs);
  }

  template<typename TYPE0, typename TYPE1>
  void add_DiagCGproduct_back0(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addBlockedCGproduct_back0Fn()(r,x,y,1,offs);
  }

  template<typename TYPE0, typename TYPE1>
  void add_DiagCGproduct_back1(const TYPE0& r, const TYPE1& x, const TYPE1& y, const int offs=0){
    SO3part_addBlockedCGproduct_back1Fn()(r,x,y,1,offs);
  }


  //template<typename VTYPE0, typename VTYPE1, typename = typename 
  //std::enable_if<std::is_base_of<SO3vec_t, VTYPE1>::value, VTYPE1>::type>
  template<typename TYPE, typename PART>
  void vCGproduct(const TYPE& r, const TYPE& x, const TYPE& y, 
    const std::function<void(const PART&, const PART&, const PART&, const int)>& lambda){
    int L=r.get_maxl();
    vector<int> offs(L+1,0);

    for(auto& p:x.parts){
      auto& P1=*p.second;
      int l1=P1.getl();
      for(auto& q:y.parts){
	auto& P2=*q.second;
	int l2=P2.getl();
	for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	  lambda(r.part(l),P1,P2,offs[l]);
	  //add_CGproduct(r.part(l),P1,P2,offs[l]);
	  offs[l]+=P1.getn()*P2.getn();
	}
      }
    }
  }

  template<typename TYPE>
  bool batches_conditionally_equal(const TYPE& x0, const TYPE& x1, const TYPE& x2){
    int t=0;
    if(x0.getb()!=1) t=x0.getb();
    if(x1.getb()!=1){
      if(t>0){if (x1.getb()!=t) return false;}
      else t=x1.getb();
    }
    if(x2.getb()!=1){
      if(t>0){if (x2.getb()!=t) return false;}
      else t=x2.getb();
    }
    return true;
  }

  inline int batches_conditionally_eq(const int b0, const int b1, const int b2){
    int t=0;
    if(b0!=1) t=b0;
    if(b1!=1){
      if(t>0){if(b1!=t) return false;}
      else t=b1;
    }
    if(b2!=1){
      if(t>0){if(b2!=t) return false;}
      else t=b2;
    }
    return true;
  }


  /*
  template<typename TYPE, typename OP, typename ROP>
  void reconcile_batches(const cnine::BatchedTensorView<TYPE>& r, const cnine::BatchedTensorView<TYPE>& x, 
    const cnine::BatchedTensorView<TYPE>& y){
    int rb=r.getb();
    int xb=x.getb();
    int yb=y.getb();
    if (!batches_conditionally_eq(rb,xb,yb)) GELIB_ERROR("Batch mismatch.");
    int B=max(rb,max(xb,yb));

    if(rb==1){
      if(xb==1 && yb==1) OP()(r,x,y);
      else ROP(r,x.cinflate(0,B).split(0,1,B),y.cinflate(0,B).split(0,1,B));
    }else{
      OP(r,x.cinflate(0,B),y.cinflate(0,B));
    }
  }
  */

  /*
  template<typename TYPE, typename OP, typename ROP>
  void reconcile_array(const cnine::TensorArrayView<TYPE>& r, const cnine::TensorArrayView<TYPE>& x, 
    const cnine::TensorArrayView<TYPE>& y){
    int rd=r.get_adims();
    int xd=x.get_adims();
    int yd=y.get_adims();

    if(rd==xd&&rd==yd){
      OP(rd.aflatten(),xd.aflatten(),yd.aflatten());
      return;
    }

    if(rd==1){
      if(xd==2&&yd==1){ // matrix-vector
	CNINE_ASSRT(r.adim(0)==x.adim(0));
	CNINE_ASSRT(y.adim(0)==x.adim(1));
	ROP(r,x,y.unsqueeze(0));
      }
      if(xd==1&&yd==2){ // vector-matrix 
	CNINE_ASSRT(r.adim(0)==y.adim(1));
	CNINE_ASSRT(x.adim(0)==y.adim(0));
	ROP(r,x.unsqueeze(0),y.permute_adims({0,1}));
      }
    }

    if(rd==2){
      if(xd==2&&yd==1){ // matrix-vector
	CNINE_ASSRT(r.adim(0)==x.adim(0));
	CNINE_ASSRT(r.adim(1)==x.adim(1));
	CNINE_ASSRT(y.adim(0)==x.adim(1));
	OP(r,x,y.insert_dim(0,x.adim(0)));
      }
      if(xd==1&&yd==2){ // vector-matrix 
	CNINE_ASSRT(r.adim(0)==y.adim(0));
	CNINE_ASSRT(r.adim(1)==y.adim(1));
	CNINE_ASSRT(x.adim(0)==y.adim(0));
	OP(r,x.insert_dim(0,y.adim(1),y));
      }
    }
  }
  */




}


#endif 
