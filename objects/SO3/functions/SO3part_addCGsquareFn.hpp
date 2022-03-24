// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addCGsquareFn
#define _SO3part_addCGsquareFn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3part3_view.hpp"
#include "MultiLoop.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

  #ifdef _WITH_CUDA
  //void SO3partB_addCGsquare_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
  //const int offs, const cudaStream_t& stream);
  #endif


  class SO3part_addCGsquareFn{
  public:

    void operator()(const SO3part3_view& _r, const SO3part3_view& _x, const int _offs=0){

      const int l=_r.getl(); 
      const int l1=_x.getl(); 
      const int diag=1-l%2;
 
      const int N1=_x.n2;
      const int B=_x.n0;
      const int dev=_r.dev;

      CNINE_CHECK_DEV2(_r,_x);
      CNINE_CHECK_BATCH2(_r,_x);

      assert(_offs+(N1*(N1-1))/2+N1*diag<=_r.n2);
      assert(l>=0 && l<=2*l1);

      if(dev==0){

	auto& C=SO3_cgbank.getf(CGindex(l1,l1,l));
	cnine::MultiLoop(B,[&](const int b){
	    SO3part2_view r=_r.slice0(b);
	    SO3part2_view x=_x.slice0(b);
	    int offs=_offs;
	    
	    for(int n1=0; n1<N1; n1++){
	      for(int n2=0; n2<n1+diag; n2++){
		for(int m1=-l1; m1<=l1; m1++){
		  for(int m2=std::max(-l1,-l-m1); m2<=std::min(l1,l-m1); m2++){
		    r.inc(m1+m2,offs+n2,C(m1+l1,m2+l1)*x(m1,n1)*x(m2,n2));
		  }
		}
	      }
	      offs+=n1+diag;
	    }
	  });

      }
      else{} //CUDA_STREAM(SO3partB_addCGsquare_cu(_r,_x,_offs,stream));

    }

  };

}

#endif
