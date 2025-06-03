/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CellFnTemplates
#define _CellFnTemplates

#include "Cnine_base.hpp"


namespace cnine{


  //typedef TensorView<complex<float> > TENSOR;
  template<typename TENSOR>
  inline void batched_mprod(const TENSOR& r, const TENSOR& x, const TENSOR& y, 
    const std::function<void(const TENSOR&, const TENSOR&, const TENSOR&)>& lambda){
    
    //CNINE_ASSRT(r.getb()==x.getb());
    //CNINE_ASSRT(r.getb()==y.getb());

    CNINE_ASSRT(r.ndims()>3);
    CNINE_ASSRT(x.ndims()>3);
    CNINE_ASSRT(y.ndims()>3);

    const int B=r.dim(0);
    const int I=r.dim(1);
    const int J=r.dim(2);
    const int K=x.dim(2);

    CNINE_ASSRT(x.dim(1)==I);
    CNINE_ASSRT(y.dim(2)==J);
    CNINE_ASSRT(y.dim(1)==K);

    MultiLoop(B,[&](const int b){
	TENSOR rb=r.slice(0,b);
	TENSOR xb=x.slice(0,b);
	TENSOR yb=y.slice(0,b);
	
	MultiLoop(I,[&](const int i){
	    MultiLoop(J,[&](const int j){
		TENSOR R=rb.slice(0,i).slice(0,j);
		for(int k=0; k<K; k++)
		  lambda(R,xb.slice(0,i).slice(0,k),yb.slice(0,k).slice(0,j));
	      });
	  });
      });
  }

}

#endif 
