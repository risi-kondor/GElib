// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addBlockedCGproductFn
#define _SO3part_addBlockedCGproductFn

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
  void SO3partB_addDiagCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream);
  #endif


  class SO3part_addBlockedCGproductFn{
  public:

    void operator()(const SO3part3_view& _r, const SO3part3_view& _x, const SO3part3_view& _y, const int bsize, const int _offs=0){

      const int l=_r.getl(); 
      const int l1=_x.getl(); 
      const int l2=_y.getl();
 
      const int N=_x.n2/bsize;
      const int N1=bsize;
      const int N2=bsize;
      const int B=_x.n0;
      const int dev=_r.dev;

      CNINE_CHECK_DEV3(_r,_x,_y)
      CNINE_CHECK_BATCH3(_r,_x,_y)

      assert(_offs+N1*N2<=_r.n2);
      assert(l>=abs(l1-l2) && l<=l1+l2);
      assert(_x.n2==_y.n2);
      assert(_x.n2%bsize==0);

      if(dev==0){

	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	cnine::MultiLoop(B,[&](const int b){
	    SO3part2_view r=_r.slice0(b);
	    SO3part2_view x=_x.slice0(b);
	    SO3part2_view y=_y.slice0(b);
	    int offs=_offs;
	    
	    for(int n=0; n<N; n++){
	      for(int n1=0; n1<N1; n1++){
		for(int n2=0; n2<N2; n2++){
		  for(int m1=-l1; m1<=l1; m1++){
		    for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		      r.inc(m1+m2,offs+n2,C(m1+l1,m2+l2)*x(m1,n1+n*bsize)*y(m2,n2+n*bsize));
		    }
		  }
		}
		offs+=N2;
	      }
	    }
	  });

      }
      else{
	assert(bsize==1);
	CUDA_STREAM(SO3partB_addDiagCGproduct_cu(_r,_x,_y,_offs,stream));
      }
    }

  };

}

#endif 
