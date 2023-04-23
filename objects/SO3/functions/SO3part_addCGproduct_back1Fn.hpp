// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addCGproduct_back1Fn
#define _SO3part_addCGproduct_back1Fn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3part3_view.hpp"
#include "MultiLoop.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

  #ifdef _WITH_CUDA
  void SO3partB_addCGproduct_back1_cu(const cnine::Ctensor3_view& yg, cnine::Ctensor3_view g, const cnine::Ctensor3_view& x, 
    const int offs, const cudaStream_t& stream);
  void SO3partB_addDiagCGproduct_back1_cu(const cnine::Ctensor3_view& yg, cnine::Ctensor3_view g, const cnine::Ctensor3_view& x, 
    const int offs, const cudaStream_t& stream);
  #endif


  class SO3part_addCGproduct_back1Fn{
  public:

    void operator()(const SO3part3_view& _yg, const SO3part3_view& _g, const SO3part3_view& _x, const int _offs=0){

      const int l=_g.getl();
      const int l1=_x.getl(); 
      const int l2=_yg.getl();
 
      const int N1=_x.n2;
      const int N2=_yg.n2;
      const int B=_x.n0;
      const int dev=_g.dev;

      CNINE_CHECK_DEV3(_yg,_g,_x);
      CNINE_CHECK_BATCH3(_yg,_g,_x);

      assert(_offs+N1*N2<=_g.n2);
      assert(l>=abs(l1-l2) && l<=l1+l2);

      LoggedTimer timer("  CGproductBack1("+to_string(l1)+","+to_string(l2)+","+to_string(l)+
	")[b="+to_string(B)+",n1="+to_string(N1)+",n2="+to_string(N2)+",dev="+to_string(dev)+"]",
	B*(2*l1+1)*(2*l2+1)*N1*N2);

      if(dev==0){

	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	cnine::MultiLoop(B,[&](const int b){
	    SO3part2_view g=_g.slice0(b);
	    SO3part2_view x=_x.slice0(b);
	    SO3part2_view yg=_yg.slice0(b);
	    int offs=_offs;

	    for(int n1=0; n1<N1; n1++){
	      for(int n2=0; n2<N2; n2++){
		for(int m1=-l1; m1<=l1; m1++){
		  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		    yg.inc(m2,n2,C(m1+l1,m2+l2)*g(m1+m2,offs+n2)*std::conj(x(m1,n1)));
		  }
		}
	      }
	      offs+=N2;
	    }
	  });

      }else{
	CUDA_STREAM(SO3partB_addCGproduct_back1_cu(_yg,_g,_x,_offs,stream));
      }

    }

  };

}

#endif
