// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addCGtransfFn
#define _SO3part_addCGtransfFn

#include "GElib_base.hpp"
//#include "CtensorB.hpp"
#include "SO3part3_view.hpp"
#include "Ctensor4_view.hpp"
#include "MultiLoop.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

  #ifdef _WITH_CUDA
  void SO3partB_addCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream);
  void SO3partB_addDiagCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream);
  #endif


  class SO3part_addCGtransfFn{
  public:


    void operator()(const SO3part3_view& _r, const cnine::Ctensor4_view& _x, const int offs=0){

      const int l=_r.getl(); 
      const int l1=(_x.n1-1)/2; 
      const int l2=(_x.n2-1)/2; 
      const int N=_x.n3;
      const int B=_x.n0;
      const int dev=_r.dev;

      CNINE_CHECK_DEV2(_r,_x);
      CNINE_CHECK_BATCH2(_r,_x);

      if(dev==0){

	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	cnine::MultiLoop(B,[&](const int b){
	    cnine::Ctensor2_view r=_r.slice0(b);
	    cnine::Ctensor3_view x=_x.slice0(b);
	    for(int i=0; i<N; i++)
	      for(int m1=-l1; m1<l1; m1++)
		for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
		  r.inc(m1+m2+l,offs+i,C(m1+l1,m2+l2)*x(m1+l1,m2+l2,i));
	  });
	  
      }else{
      }

    }

    

  };

}

#endif 
