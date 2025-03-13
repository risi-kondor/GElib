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


namespace GElib{

#ifdef _WITH_CUDA
  void SO3part_addCGproduct_cu(SO3part r, SO3part x, SO3part y, const int offs, const cudaStream_t& stream);
#endif


  template<typename PART, typename TYPE>
  class SO3part_addCGproductFn{
  public:

    //typedef SO3part<TYPE> PART;
    typedef cnine::TensorView<complex<TYPE> > TENSOR;

    void operator()(PART r, PART x, PART y, const int _offs=0){
      const int l=r.getl(); 
      const int l1=x.getl(); 
      const int l2=y.getl();
      const int N1=x.getn();
      const int N2=y.getn();

      const int dev=r.dev;
      GELIB_ASSRT(x.get_dev()==dev);
      GELIB_ASSRT(x.get_dev()==dev);

      r.reconcile_batches(x,y);
      r.reconcile_grids(x,y);
      r.cocanonicalize_to_5d(x,y);

      if(dev==0){
	auto& C=SO3_CGbank.get<TYPE>(l1,l2,l);
	r.for_each_cell_multi(x,y,[&](const TENSOR& _r, const TENSOR& _x, const TENSOR& _y){
	    int offs=_offs;
	    for(int n1=0; n1<N1; n1++){
	      for(int n2=0; n2<N2; n2++){
		for(int m1=-l1; m1<=l1; m1++){
		  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		    _r.inc(m1+m2+l,offs+n2,C(m1+l1,m2+l2)*_x(m1+l1,n1)*_y(m2+l2,n2));
		  }
		}
	      }
	      offs+=N2;
	    }
	  });
      }

      if(dev==1){
	CUDA_STREAM(SO3part_addCGproduct_cu(r,x,y,_offs,stream));
      }

    }


  };

}


#endif 



	/*
	r.for_each_cell_multi(x,y,[&](const int b, const int g, const TENSOR& _r, const TENSOR& _x, const TENSOR& _y){
	    int offs=_offs;
	    for(int n1=0; n1<N1; n1++){
	      for(int n2=0; n2<N2; n2++){
		for(int m1=-l1; m1<=l1; m1++){
		  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		    _r.inc(m1+m2+l,offs+n2,C(m1+l1,m2+l2)*_x(m1+l1,n1)*_y(m2+l2,n2));
		  }
		}
	      }
	      offs+=N2;
	    }
	  });
	*/
