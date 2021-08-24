
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partA_CGproduct_back0_cop
#define _SO3partA_CGproduct_back0_cop

#include "GenericOp.hpp"
#include "SO3_CGbank.hpp"

extern GElib::SO3_CGbank SO3_cgbank;


namespace GElib{

  class SO3partArrayA;

#ifdef _WITH_CUDA
  template<typename CMAP>
  void SO3partA_CGproduct_back0_cu(const CMAP& map, SO3partArrayA& r, const SO3partArrayA& g, 
    const SO3partArrayA& y, const cudaStream_t& stream, const int offs, const int mode);
#endif 


  class SO3partA_CGproduct_back0_cop{
  public:

    int offs;
    int l=0;

    SO3partA_CGproduct_back0_cop(const int _offs=0): offs(_offs){}

    SO3partA_CGproduct_back0_cop(const int _l, const int _offs): offs(_offs), l(_l){}

  public:

    virtual void apply(SO3partA& r, const SO3partA& g, const SO3partA& y, const int add_flag=0) const{
    
      int off=offs;
      const int l=g.getl(); 
      const int l1=r.getl(); 
      const int l2=y.getl(); 
      const int N1=r.getn();
      const int N2=y.getn();
      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));

      if(add_flag==0) r.set_zero();

      if(r.nbu==-1){
	for(int n1=0; n1<N1; n1++){
	  for(int n2=0; n2<N2; n2++){
	    for(int m1=-l1; m1<=l1; m1++)
	      for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		r.inc(n1,m1+l1,C(m1+l1,m2+l2)*std::conj(y(n2,m2+l2))*g(off+n2,m1+m2+l));
	      }
	  }
	  off+=N2;
	}
	return;
      }
    }

    
    template<typename IMAP>
    void apply(const IMAP& map, SO3partArrayA& r, const SO3partArrayA& g, const SO3partArrayA& y,
      const int add_flag) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      SO3partA_CGproduct_back0_cu(map,r,g,y,stream,offs,1-add_flag);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      CNINE_NOCUDA_ERROR;
#endif
    }

    template<typename IMAP>
    void accumulate(const IMAP& map, SO3partArrayA& r, const SO3partArrayA& g, const SO3partArrayA& y,
      const int add_flag) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      SO3partA_CGproduct_back0_cu(map,r,g,y,stream,offs,2);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      CNINE_NOCUDA_ERROR;
#endif
    }

  };

}

#endif 


