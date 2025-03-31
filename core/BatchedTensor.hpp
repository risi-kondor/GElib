/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _BatchedTensor
#define _BatchedTensor

#include "GElib_base.hpp"
#include "TensorView.hpp"


namespace cnine{

  template<typename TYPE>
  class BatchedTensor: public cnine::TensorView<TYPE>{
  public:
  
    using TENSOR=cnine::TensorView<TYPE>;

    using TENSOR::dims;
    using TENSOR::slice;
    using TENSOR::repr;

    using TENSOR::TENSOR;


    BatchedTensor(const TENSOR& x):
      TENSOR(x){}


  public: // ---- Batches ------------------------------------------------------------------------------------


    bool is_batched() const{
      return dims[0]>1;
    }

    int getb() const{
      return dims[0];
    }

    int dominant_batch(const BatchedTensor& y) const{
      int xb=getb();
      int yb=y.getb();
      if(xb==yb) return xb;
      if(xb==1) return yb;
      if(yb==1) return xb;
      throw std::invalid_argument("Cnine error: the batch dimensions of "+repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return 0;
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------

    
    template<typename TYPE2>
    void for_each_batch_multi(const cnine::TensorView<TYPE2>& x,
      const std::function<void(const int, const TENSOR& r, const cnine::TensorView<TYPE2>& x)>& lambda) const{
      
      if(getb()==1){
	int B=x.dim(0);
	for(int b=0; b<B; b++)
	  lambda(b,slice(0,0),x.slice(0,b));
	return;
      }

      if(x.dim(0)==1){
	cnine::MultiLoop(getb(),[&](const int b){
	    lambda(b,slice(0,b),x.slice(0,0));
	  });
      }

      cnine::MultiLoop(getb(),[&](const int b){
	  lambda(b,slice(0,b),x.slice(0,b));
	});
     }


    void for_each_batch_multi(const BatchedTensor& x, const BatchedTensor& y,
      const std::function<void(const int, const TENSOR& r, const TENSOR& x, const TENSOR& y)>& lambda) const{
      int B=getb();
      GELIB_ASSRT(x.getb()==B);
      GELIB_ASSRT(y.getb()==B);
      cnine::MultiLoop(B,[&](const int b){
	  lambda(b,slice(0,b),x.slice(0,b),y.slice(0,b));
	});
    }

  };

}

#endif 
