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

#ifndef _Btensor_add_RprodFn
#define _Btensor_add_RprodFn

#include "Cnine_base.hpp"
#include "BatchedTensorView.hpp"


namespace cnine{

  #ifdef _WITH_CUDA
  #endif

  template<typename TYPE>
  class BatchedTensorView;

  class Btensor_add_RprodFn{
  public:

    template<typename TYPE> // TODO 
    void operator()(const BatchedTensorView<TYPE>& _r, 
      const BatchedTensorView<TYPE>& _x, const BatchedTensorView<TYPE>& _y){

      int dev=_r.dev;

      if(dev==0){
	_r.for_each_batch(_x,_y,[&](const int b, const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){
	    r.add_prod(x,y);
	  });

      }else{
	CNINE_UNIMPL();
      }

    }

  };

}

#endif 
