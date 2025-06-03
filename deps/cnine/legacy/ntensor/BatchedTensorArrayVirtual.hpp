/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineBatchedTensorArrayVirtual
#define _CnineBatchedTensorArrayVirtual

#include "Cnine_base.hpp"
#include "TensorArrayView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 

// deprecated 
namespace cnine{

  template<typename TYPE, typename BASE>
  class BatchedTensorArrayVirtual: public BASE{
  public:

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;
    using BASE::ndims;


  public: // ---- Constructors ------------------------------------------------------------------------------


    BatchedTensorArrayVirtual(){};

    //BatchedTensorArrayVirtual(const BatchedTensorArrayVirtual<TYPE,BASE>& x):
    //BatchedTensorArrayVirtual(x.getb(),x.adims(),x.ddims(),x.dev){
    //CNINE_COPY_WARNING();
    //view()=x.view();
    //}


  public: // ---- Named constructors ------------------------------------------------------------------------


  public: // ---- Copying -----------------------------------------------------------------------------------


    BatchedTensorArrayVirtual(const BatchedTensorArrayVirtual& x):
      BatchedTensorArrayVirtual(x.dims,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    BatchedTensorArrayVirtual(const BatchedTensorArrayVirtual& x, const nowarn_flag& dummy):
      BatchedTensorArrayVirtual(x.dims,x.dev){
      view()=x.view();
    }
        
    BatchedTensorArrayVirtual(const BatchedTensorArrayVirtual&& x):
      BASE(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
    }
        
    BatchedTensorArrayVirtual& operator=(const BatchedTensorArrayVirtual& x){
      arr=x.arr;
      return *this;
    }

    
  public: // ---- Conversions ---------------------------------------------------------------------------------


    BatchedTensorArrayVirtual(const BASE& x):
      BatchedTensorArrayVirtual(x.get_adims(),x.get_ddims(),x.dev){
      CNINE_CONVERT_WARNING();
      view()=x;
    }


  public: // ---- Transport -----------------------------------------------------------------------------------


    BatchedTensorArrayVirtual(const BASE& x, const int _dev):
      BatchedTensorArrayVirtual(x.dims,_dev){
      CNINE_COPY_WARNING();
      view()=x;
    }

    void move_to_device(const int _dev) const{
      if(dev==_dev) return;
      const_cast<BatchedTensorArrayVirtual&>(*this)=BatchedTensorArrayVirtual(*this,_dev);
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    BatchedTensorArrayVirtual(const at::Tensor& T):
      BatchedTensorArrayVirtual(Gdims(T),T.type().is_cuda()){
      (*this)=T;
    }

    #endif


  public: // ---- Views -------------------------------------------------------------------------------------


    BASE view(){
      return BASE(*this);
    }

    const BASE view() const{
      return BASE(*this);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    /*
    BatchedTensorArrayVirtual operator*(const BASE& y) const{
      CNINE_ASSERT(ndims()==1||ndims()==2,"first operand of product must be a vector or a matrix");
      CNINE_ASSERT(y.ndims()==1||y.ndims()==2,"second operand of product must be a vector or a matrix");

      if(ndims()==1 && y.ndims()==2){
	BatchedTensorArrayVirtual R=zero({y.dims[1]},dev);
	R.add_mvprod_T(y,*this);
	return R;
      }

      if(ndims()==2 && y.ndims()==1){
	BatchedTensorArrayVirtual R=zero({dims[0]},dev);
	R.add_mvprod(*this,y);
	return R;
      }

      if(ndims()==2 && y.ndims()==2){
	BatchedTensorArrayVirtual R=zero({dims[0],y.dims[1]},dev);
	R.add_mprod(*this,y);
	return R;
      }

      return BatchedTensorArrayVirtual();
    }
    */


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "BatchedTensorArrayVirtual";
    }

  };

}
    
#endif 


