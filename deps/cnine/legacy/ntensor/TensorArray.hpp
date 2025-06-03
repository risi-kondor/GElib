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


#ifndef _CnineTensorArray
#define _CnineTensorArray

#include "Cnine_base.hpp"
#include "TensorArrayVirtual.hpp"
#include "Tensor.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE>
  class TensorArray: public TensorArrayVirtual<TYPE,TensorArrayView<TYPE> >{
  public:

    typedef TensorArrayVirtual<TYPE,TensorArrayView<TYPE> > TensorArrayVirtual;

    using TensorArrayVirtual::TensorArrayVirtual;
    using TensorArrayVirtual::arr;
    using TensorArrayVirtual::move_to_device;

    ~TensorArray(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    TensorArray(const Gdims& _adims, const Gdims& _ddims, const int _dev=0):
      TensorArrayVirtual(_adims,_ddims,_dev){}


  public: // ---- Named constructors --------------------------------------------------------------------------


    static TensorArray zero(const Gdims& _adims, const Gdims& _ddims, const int _dev=0){
      return TensorArray(_adims,_ddims,cnine::fill_zero(),_dev);}
    
    static TensorArray sequential(const Gdims& _adims, const Gdims& _ddims, const int _dev=0){
      return TensorArray(_adims,_ddims,cnine::fill_sequential(),_dev);}
    
    static TensorArray gaussian(const Gdims& _adims, const Gdims& _ddims, const int _dev=0){
      return TensorArray(_adims,_ddims,cnine::fill_gaussian(),_dev);}
    

  public: // ---- Conversions ---------------------------------------------------------------------------------


    //unnecessary
    //TensorArray(const TensorArrayVirtual& x):
    //TensorArrayVirtual(x){}


  public: // ---- Views -------------------------------------------------------------------------------------


    TensorArray(const TensorArrayView<TYPE>& x):
      TensorArray(x.adims(),x.ddims(),x.dev){
      CNINE_COPY_WARNING();
      view()=x;

    }

    TensorArrayView<TYPE> view(){
      return TensorArrayView<TYPE>(*this);
    }

    const TensorArrayView<TYPE> view() const{
      return TensorArrayView<TYPE>(*this);
    }


  public: // ---- Access --------------------------------------------------------------------------------------


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorArray";
    }



  };


}

#endif 
