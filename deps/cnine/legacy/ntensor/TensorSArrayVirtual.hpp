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


#ifndef _CnineTensorSArrayVirtual
#define _CnineTensorSArrayVirtual

#include "Cnine_base.hpp"
#include "TensorSArrayView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE, typename BASE>
  class TensorSArrayVirtual: public BASE{
  public:

    using BASE::BASE;
    using BASE::arr;
    using BASE::offs;
    using BASE::ddims;
    using BASE::dstrides;


  public: // ---- Constructors ------------------------------------------------------------------------------


    //TensorSArrayVirtual(){};


  public: // ---- Named constructors ------------------------------------------------------------------------


    static TensorSArrayVirtual zero(const Gdims& _adims, const Gdims& _ddims, const SparseTensor<int>& _offs, const int _dev=0){
      return TensorSArrayVirtual(_adims,_ddims,_offs,cnine::fill_zero(),_dev);}
    
    static TensorSArrayVirtual sequential(const Gdims& _adims, const Gdims& _ddims, const SparseTensor<int>& _offs, const int _dev=0){
      return BASE(_adims,_ddims,_offs,cnine::fill_sequential(),_dev);}
    
    static TensorSArrayVirtual gaussian(const Gdims& _adims, const Gdims& _ddims, const SparseTensor<int>& _offs, const int _dev=0){
      return TensorSArrayVirtual(_adims,_ddims,_offs,cnine::fill_gaussian(),_dev);}
    


  public: // ---- Copying -----------------------------------------------------------------------------------


    TensorSArrayVirtual(const TensorSArrayVirtual& x):
      TensorSArrayVirtual(x.dims,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    TensorSArrayVirtual(const TensorSArrayVirtual& x, const nowarn_flag& dummy):
      TensorSArrayVirtual(x.dims,x.dev){
      view()=x.view();
    }
        
    TensorSArrayVirtual(const TensorSArrayVirtual&& x):
      BASE(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
    }
        
    TensorSArrayVirtual& operator=(const TensorSArrayVirtual& x){
      arr=x.arr;
      return *this;
    }

    
  public: // ---- Conversions ---------------------------------------------------------------------------------


    TensorSArrayVirtual(const BASE& x):
      TensorSArrayVirtual(x.get_adims(),x.get_ddims(),x.dev){
      CNINE_CONVERT_WARNING();
      view()=x;
    }


  public: // ---- Transport -----------------------------------------------------------------------------------


    TensorSArrayVirtual(const BASE& x, const int _dev):
      TensorSArrayVirtual(x.dims,_dev){
      CNINE_COPY_WARNING();
      view()=x;
    }

    void move_to_device(const int _dev) const{
      //if(dev==_dev) return;
      const_cast<TensorSArrayVirtual&>(*this)=TensorSArrayVirtual(*this,_dev);
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    //TensorSArrayVirtual(const at::Tensor& T):
    //TensorSArrayVirtual(Gdims(x),T.type().is_cuda()){
    //(*this)=T;
    //}

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
    TensorSArrayVirtual operator*(const BASE& y) const{
      CNINE_ASSERT(ndims()==1||ndims()==2,"first operand of product must be a vector or a matrix");
      CNINE_ASSERT(y.ndims()==1||y.ndims()==2,"second operand of product must be a vector or a matrix");

      if(ndims()==1 && y.ndims()==2){
	TensorSArrayVirtual R=zero({y.dims[1]},dev);
	R.add_mvprod_T(y,*this);
	return R;
      }

      if(ndims()==2 && y.ndims()==1){
	TensorSArrayVirtual R=zero({dims[0]},dev);
	R.add_mvprod(*this,y);
	return R;
      }

      if(ndims()==2 && y.ndims()==2){
	TensorSArrayVirtual R=zero({dims[0],y.dims[1]},dev);
	R.add_mprod(*this,y);
	return R;
      }

      return TensorSArrayVirtual();
    }
    */


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorSArrayVirtual";
    }

  };

}
    
#endif 


