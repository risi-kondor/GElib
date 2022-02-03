// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addCGproductFn
#define _SO3part_addCGproductFn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3part3_view.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

#ifdef _WITH_CUDA
  void SO3partB_addCGproduct_cu(cnine::Ctensor2_view r, const cnine::Ctensor2_view& x, const cnine::Ctensor2_view& y, 
    const cudaStream_t& stream, const int offs=0);
#endif


  class SO3part_addCGproductFn{
  public:


    void operator()(SO3part3_view& _r, const SO3part3_view& _x, const SO3part3_view& _y, const int _offs=0){

      const int l=_r.getl(); 
      const int l1=_x.getl(); 
      const int l2=_y.getl();
 
      const int N1=_x.n2;
      const int N2=_y.n2;
      const int B=_x.n0;

      const int dev=_r.dev;
      assert(_x.dev==dev);
      assert(_y.dev==dev);

      assert(_y.n0==B);
      assert(_r.n0==B);
      assert(_offs+N1*N2<=_r.n2);
      assert(l>=abs(l1-l2) && l<=l1+l2);

      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));

      if(dev==0){
        for(int b=0; b<B; b++){

	  SO3part2_view r=_r.slice0(b);
	  SO3part2_view x=_x.slice0(b);
	  SO3part2_view y=_y.slice0(b);
	  int offs=_offs;

	  for(int n1=0; n1<N1; n1++){
	    for(int n2=0; n2<N2; n2++){
	      for(int m1=-l1; m1<=l1; m1++){
		for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		  //cout<<"   "<<n1<<" "<<n2<<" "<<m1<<" "<<m2<<endl;
		  r.inc(m1+m2,offs+n2,C(m1+l1,m2+l2)*x(m1,n1)*y(m2,n2));
		}
	      }
	    }
	    offs+=N2;
	  }
        }
      }else{

#ifdef _WITH_CUDA
	assert(_x.dev==1);
	assert(_y.dev==1);
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
        for(int b=0; b<B; b++){
	  SO3part2_view r=_r.slice0(b);
	  SO3part2_view x=_x.slice0(b);
	  SO3part2_view y=_y.slice0(b);
	  SO3partB_addCGproduct_cu(r,x,y,stream,_offs);
        }
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
