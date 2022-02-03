// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addCGproduct_back0Fn
#define _SO3part_addCGproduct_back0Fn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3part3_view.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

#ifdef _WITH_CUDA
  void SO3partB_addCGproduct_back0_cu(cnine::Ctensor3_view xg, const cnine::Ctensor3_view& rg, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream);
#endif

  class SO3part_addCGproduct_back0Fn{
  public:


    void operator()(SO3part3_view& _xg, const SO3part3_view& _g, const SO3part3_view& _y, const int _offs=0){

      const int l=_g.getl(); 
      const int l1=_xg.getl(); 
      const int l2=_y.getl();
 
      const int N1=_xg.n2;
      const int N2=_y.n2;
      const int B=_xg.n0;

      const int dev=_g.dev;
      assert(_x.dev==dev);
      assert(_y.dev==dev);

      assert(_y.n0==B);
      assert(_g.n0==B);
      assert(_offs+N1*N2<=_g.n2);
      assert(l>=abs(l1-l2) && l<=l1+l2);

      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));

      if(dev==0){
	for(int b=0; b<B; b++){
	  SO3part2_view g=_g.slice0(b);
	  SO3part2_view xg=_xg.slice0(b);
	  SO3part2_view y=_y.slice0(b);
	  int offs=_offs;
	  for(int n1=0; n1<N1; n1++){
	    for(int n2=0; n2<N2; n2++){
	      for(int m1=-l1; m1<=l1; m1++){
		for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		  xg.inc(m1,n1,C(m1+l1,m2+l2)*g(m1+m2,offs+n2)*std::conj(y(m2,n2)));
		}
	      }
	    }
	    offs+=N2;
	  }
	}
      }else{
#ifdef _WITH_CUDA
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	SO3partB_addCGproduct_back0_cu(_xg,_g,_y,_offs,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif

      }

    }
    


  };


}

#endif
