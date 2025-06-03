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


#ifndef _CnineTensorSArray
#define _CnineTensorSArray

#include "Cnine_base.hpp"
#include "TensorSArrayVirtual.hpp"
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
  class TensorSArray: public TensorSArrayVirtual<TYPE,TensorSArrayView<TYPE> >{
  public:

    typedef TensorSArrayVirtual<TYPE,TensorSArrayView<TYPE> > TensorSArrayVirtual;

    using TensorSArrayVirtual::TensorSArrayVirtual;
    using TensorSArrayVirtual::arr;
    using TensorSArrayVirtual::move_to_device;


  public: // ---- Constructors --------------------------------------------------------------------------------


    TensorSArray(const Gdims& _adims, const Gdims& _ddims, const int _dev=0):
      TensorSArrayVirtual(_adims,_ddims,_dev){}


  public: // ---- Named constructors --------------------------------------------------------------------------


    static TensorSArray sequential(const Gdims& _adims, const Gdims& _ddims, const map_of_lists<int,int>& mask, const int _dev=0){
      return TensorSArray(_adims,_ddims,mask,cnine::fill_sequential(),_dev);}
    

    static TensorSArray zero(const Gdims& _adims, const Gdims& _ddims, const int _dev=0){
      return TensorSArray(_adims,_ddims,cnine::fill_zero(),_dev);}
    
    static TensorSArray sequential(const Gdims& _adims, const Gdims& _ddims, const SparseTensor<int>& mask, const int _dev=0){
      return TensorSArray(_adims,_ddims,mask,cnine::fill_sequential(),_dev);}
    
    static TensorSArray gaussian(const Gdims& _adims, const Gdims& _ddims, const int _dev=0){
      return TensorSArray(_adims,_ddims,cnine::fill_gaussian(),_dev);}
    

  public: // ---- Conversions ---------------------------------------------------------------------------------


  public: // ---- Access --------------------------------------------------------------------------------------


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorSArray";
    }



  };


}

#endif 
