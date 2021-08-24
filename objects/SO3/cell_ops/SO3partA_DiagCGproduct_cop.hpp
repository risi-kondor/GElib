
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partA_DiagCGproduct_cop
#define _SO3partA_DiagCGproduct_cop

#include "GenericOp.hpp"
#include "SO3_CGbank.hpp"

extern GElib::SO3_CGbank SO3_cgbank;


namespace GElib{

  class SO3partArrayA;

#ifdef _WITH_CUDA

  template<typename CMAP>
  void SO3partA_DiagCGproduct_cu(const CMAP& map, SO3partArrayA& r, const SO3partArrayA& x, 
    const SO3partArrayA& y, const cudaStream_t& stream, const int offs, const int mode);

#endif 



  class SO3partA_DiagCGproduct_cop{
  public:

    int offs;
    int l=0;

    SO3partA_DiagCGproduct_cop(const int _offs=0): offs(_offs){}
    
    SO3partA_DiagCGproduct_cop(const int _l, const int _offs): offs(_offs), l(_l){}

  public:


    void apply(SO3partA& r, const SO3partA& x, const SO3partA& y, const int add_flag=0) const{
    
      //int off=offs;
      const int l=r.getl(); 
      const int l1=x.getl(); 
      const int l2=y.getl(); 
      const int N=x.getn();
      assert(y.getn()==N);
      assert(r.getn()==N);
      const int N2=y.getn();
      auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
   
      if(add_flag==0) r.set_zero();

      for(int n=0; n<N; n++)
	for(int m1=-l1; m1<=l1; m1++)
	  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++)
	    r.inc(offs+n,m1+m2+l,C(m1+l1,m2+l2)*x(n,m1+l1)*y(n,m2+l2));
      
    }


    template<typename IMAP>
    void apply(const IMAP& map, SO3partArrayA& r, const SO3partArrayA& x, const SO3partArrayA& y, 
      const int add_flag) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      SO3partA_DiagCGproduct_cu(map,r,x,y,stream,offs,1-add_flag);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }

    template<typename IMAP>
    void accumulate(const IMAP& map, SO3partArrayA& r, const SO3partArrayA& x, const SO3partArrayA& y, 
      const int add_flag=0) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      SO3partA_DiagCGproduct_cu(map,r,x,y,stream,offs,2);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }

  };

}

#endif 


 
