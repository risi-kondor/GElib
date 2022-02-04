// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Fpart_addFmodsq_Fn
#define _SO3Fpart_addFmodsq_Fn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3Fpart3_view.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

#ifdef _WITH_CUDA
  void SO3Fpart_addFmodsq_cu(cnine::Ctensor3_view& r, const cnine::Ctensor3_view x, const cnine::Ctensor3_view& y, 
    const cudaStream_t& stream);
#endif


  class SO3Fpart_addFmodsq_Fn{
  public:

    void operator()(SO3Fpart3_view& _r, const SO3Fpart3_view& _x, const SO3Fpart3_view& _y){

      const int l=_r.getl(); 
      const int l1=_x.getl(); 
      const int l2=_y.getl();
      assert(l>=abs(l1-l2) && l<=l1+l2);

      const int B=_r.n0;
      assert(_x.n0==B);
      assert(_y.n0==B);

      const int dev=_r.dev;
      assert(_x.dev==dev);
      assert(_y.dev==dev);

      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
      const float c=((2.0*l1+1)+(2.0*l2+1))/(2.0*l+1);

      if(dev==0){
	for(int b=0; b<B; b++){
	  SO3Fpart2_view r=_r.slice0(b);
	  SO3Fpart2_view x=_x.slice0(b);
	  SO3Fpart2_view y=_y.slice0(b).flip();	
	  for(int M1=-l1; M1<=l1; M1++){
	    for(int M2=std::max(-l2,-l-M1); M2<=std::min(l2,l-M1); M2++){
	      float t=C(M1+l1,M2+l2)*c;
	      for(int m1=-l1; m1<=l1; m1++){
		for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		  //cout<<"   "<<n1<<" "<<n2<<" "<<m1<<" "<<m2<<endl;
		  r.inc(M1+M2,m1+m2,t*C(m1+l1,m2+l2)*x(M1,m1)*std::conj(y(M2,m2)));
		}
	      }
	    }
	  }
	}
      }else{
#ifdef _WITH_CUDA
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	SO3Fpart_addFmodsq_cu(_r,_x,_y,stream);
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
