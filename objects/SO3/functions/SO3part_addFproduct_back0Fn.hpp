// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addFproduct_back0Fn
#define _SO3part_addFproduct_back0Fn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
//#include "SO3Fpart3_view.hpp"
#include "Ctensor3_view.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

  #ifdef _WITH_CUDA
  void SO3Fpart_addFproduct_back0_cu(const cnine::Ctensor3_view& xg, const cnine::Ctensor3_view& g, 
  const cnine::Ctensor3_view& y, const int conj, 
    const cudaStream_t& stream);
  #endif


  class SO3part_addFproduct_back0Fn{
  public:

    int conj=0;

    SO3part_addFproduct_back0Fn(){}
    SO3part_addFproduct_back0Fn(const int _conj): conj(_conj){}

  public:

    void operator()(const cnine::Ctensor3_view& _xg, const cnine::Ctensor3_view& _g, const cnine::Ctensor3_view& _y){

      const int l=(_g.n1-1)/2;
      const int l1=(_xg.n1-1)/2; 
      const int l2=(_y.n1-1)/2;
      const int B=_xg.n0;
      const int dev=_g.dev;

      CNINE_CHECK_DEV3(_g,_xg,_y)
      CNINE_CHECK_BATCH3(_g,_xg,_y)
      assert(l>=abs(l1-l2) && l<=l1+l2);

      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
      const float c=((2.0*l1+1)*(2.0*l2+1))/(2.0*l+1);

      if(dev==0)
	cnine::MultiLoop(B,[&](const int b){
	    SO3Fpart2_view g=_g.slice0(b);
	    SO3Fpart2_view xg=_xg.slice0(b);
	    SO3Fpart2_view y=_y.slice0(b);

	    if(conj%2==0){
	      for(int M1=-l1; M1<=l1; M1++){
		for(int M2=std::max(-l2,-l-M1); M2<=std::min(l2,l-M1); M2++){
		  float t=C(M1+l1,M2+l2)*c;
		  for(int m1=-l1; m1<=l1; m1++){
		    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		      xg.inc(M1,m1,t*C(m1+l1,m2+l2)*g(M1+M2,m1+m2)*std::conj(y(M2,m2)));
		    }
		  }
		}
	      }
	    }else{
	      for(int M1=-l1; M1<=l1; M1++){
		for(int M2=std::max(-l2,-l-M1); M2<=std::min(l2,l-M1); M2++){
		  float t=C(M1+l1,M2+l2)*c;
		  for(int m1=-l1; m1<=l1; m1++){
		    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		      xg.inc(M1,m1,t*C(m1+l1,m2+l2)*g(M1+M2,m1+m2)*y(M2,m2));
		    }
		  }
		}
	      }
	    }
	  });
      else
	CUDA_STREAM(SO3Fpart_addFproduct_back0_cu(_xg,_g,_y,conj,stream));

    }
    
  };


}

#endif
