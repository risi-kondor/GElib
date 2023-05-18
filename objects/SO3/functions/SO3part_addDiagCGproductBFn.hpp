// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addDiagCGproductBFn
#define _SO3part_addDiagCGproductBFn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3part3_view.hpp"
#include "SO3part4_view.hpp"
#include "MultiLoop.hpp"
#include "GElibTimer.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

  #ifdef _WITH_CUDA
  void Ctensor2_add_otimes_cu(const Ctensor2_view& r, const Ctensor2_view& x, const Ctensor2_view& y, const float c,
    const cudaStream_t& stream);
  #endif


  class SO3part_addDiagCGproductBFn{
  public:

    void operator()(const SO3part3_view& _r, const SO3part3_view& _x, const SO3part3_view& _y, const int _offs=0){

      const int l=_r.getl(); 
      const int l1=_x.getl(); 
      const int l2=_y.getl();
 
      const int N=_x.n2;
      const int B=_x.n0;
      const int dev=_r.dev;

      CNINE_CHECK_DEV3(_r,_x,_y)
      CNINE_CHECK_BATCH3(_r,_x,_y)
      assert(l>=abs(l1-l2) && l<=l1+l2);
      assert(_x.n2==_y.n2);

      if(dev==0){
	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	cnine::MultiLoop(B,[&](const int b){
	    SO3part2_view r=_r.slice0(b);
	    SO3part2_view x=_x.slice0(b);
	    SO3part2_view y=_y.slice0(b);
	    
	    for(int m1=-l1; m1<=l1; m1++){
	      for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		for(int n=0; n<N; n++){
		  r.inc(m1+m2,_offs+n,C(m1+l1,m2+l2)*x(m1,n)*y(m2,n));
		}
	      }
	    }
	  });
      }
      else{
#ifdef _WITH_CUDA
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	for(int m1=-l1; m1<=l1; m1++){
	  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
	    cnine::Ctensor2_view r(_r.arr+r.s1*(m1+m2-l)+_offs*r.s2,_r.arrc+r.s1*(m1+m2-l)+_offs*r.s2,
	      _r.n0,_r.n2,_r.s0,_r.s2,_r.dev);
	    cnine::Ctensor2_view x(_x.arr+x.s1*(m1-l1),_x.arrc+x.s1*(m1-l1),_x.n0,_x.n2,_x.s0,_x.s2,_x.dev);
	    cnine::Ctensor2_view y(_y.arr+y.s1*(m2-l2),_y.arrc+x.s2*(m2-l2),_y.n0,_y.n2,_y.s0,_y.s2,_y.dev);
	    cnine::Ctensor2_add_otimes_cu(r,x,y,C(m1+l1,m2+l2),stream);
	  }
	}
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#endif
      }
    }

  };

}

#endif 
