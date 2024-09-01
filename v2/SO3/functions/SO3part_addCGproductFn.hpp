// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2024, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3part_addCGproductFn
#define _SO3part_addCGproductFn

#include "GElib_base.hpp"
#include "SO3part.hpp"
#include "SO3CGbank.hpp"
#include "MultiLoop.hpp"
//#include "GElibTimer.hpp"
#include "WorkStreamLoop.hpp"

//extern thread_local cnine::DeviceSelector cnine::dev_selector;

extern GElib::SO3CGbank SO3_CGbank;
//extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

  #ifdef _WITH_CUDA
  //void SO3partB_addCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
  //const int offs, const cudaStream_t& stream);
  //void SO3partB_addDiagCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
  //const int offs, const cudaStream_t& stream);
  #endif


  template<typename PART, typename TYPE>
  class SO3part_addCGproductFn{
  public:

    //typedef SO3part<TYPE> PART;
    typedef cnine::TensorView<complex<TYPE> > TENSOR;

    void operator()(const PART& r, const PART& x, const PART& y, const int _offs=0){
      if(r.ndims()>3) gridded(r,x,y);

      GELIB_ASSRT(r.ndims()==3);
      GELIB_ASSRT(x.ndims()==3);
      GELIB_ASSRT(y.ndims()==3);

      const int B=r.getb();
      GELIB_ASSRT(x.getb()==B);
      GELIB_ASSRT(y.getb()==B);

      const int l=r.getl(); 
      const int l1=x.getl(); 
      const int l2=y.getl();
 
      const int N1=x.getn();
      const int N2=y.getn();

      const int dev=r.dev;
      GELIB_ASSRT(x.get_dev()==dev);
      GELIB_ASSRT(x.get_dev()==dev);

      if(dev==0){
	auto& C=SO3_CGbank.get<TYPE>(l1,l2,l);
	r.for_each_batch_multi(x,y,[&](const int b, const TENSOR& r, const TENSOR& x, const TENSOR& y){
	    int offs=_offs;
	    for(int n1=0; n1<N1; n1++){
	      for(int n2=0; n2<N2; n2++){
		for(int m1=-l1; m1<=l1; m1++){
		  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		    r.inc(m1+m2+l,offs+n2,C(m1+l1,m2+l2)*x(m1+l1,n1)*y(m2+l2,n2));
		  }
		}
	      }
	      offs+=N2;
	    }
	    cout<<r<<endl;
	  });
      }

    }


  private: // ---- Gridded -----------------------------------------------------------------------------------

    void gridded(const PART& r, const PART& x, const PART& y){
    }

  };

}


#endif 
