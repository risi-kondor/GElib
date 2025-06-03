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


#ifndef _Cnine_TensorView_functions
#define _Cnine_TensorView_functions

#include "TensorView.hpp"
#include "Einsum1.hpp"
#include "Einsum2.hpp"


namespace cnine{


  // ---- Addition -------------------------------------------------------------------------------------------

  template<typename TYPE, typename TYPE2>
  TensorView<TYPE> operator+(const TYPE2 c, const TensorView<TYPE>& x){
    TensorView<TYPE> r=x.copy();
    r.add(c);
    return r;
  }

  template<typename TYPE, typename TYPE2>
  TensorView<TYPE> operator+(const TensorView<TYPE>& x, const TYPE2 c){
    TensorView<TYPE> r=x.copy();
    r.add(c);
    return r;
  }

  template<typename TYPE>
  TensorView<TYPE> operator+(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    TensorView<TYPE> r=x.copy();
    r.add(y);
    return r;
  }


  // ---- Subtraction ----------------------------------------------------------------------------------------

  
  template<typename TYPE>
  TensorView<TYPE> operator-(const TensorView<TYPE>& x){
   TensorView<TYPE> r=x.zeros_like();
    r.add(x,-1);
    return r;
  }

  template<typename TYPE, typename TYPE2>
  TensorView<TYPE> operator-(const TYPE2 c, const TensorView<TYPE>& x){
    TensorView<TYPE> r=x.zeros_like();
    r.add(x,-1);
    r.add(c);
    return r;
  }

  template<typename TYPE, typename TYPE2>
  TensorView<TYPE> operator-(const TensorView<TYPE>& x, const TYPE2 c){
    TensorView<TYPE> r=x.copy();
    r.add(-c);
    return r;
  }

  template<typename TYPE>
  TensorView<TYPE> operator-(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    TensorView<TYPE> r=x.copy();
    r.subtract(y);
    return r;
  }


  // ---- Multiplication -------------------------------------------------------------------------------------

  
  template<typename TYPE, typename TYPE2>
  TensorView<TYPE> operator*(const TensorView<TYPE>& x, const TYPE2 c){
    TensorView<TYPE> r=x.zeros_like();
    r.add(x,c);
    return r;
  }

  template<typename TYPE, typename TYPE2>
  TensorView<TYPE> operator*( const TYPE2 c, const TensorView<TYPE>& x){
    TensorView<TYPE> r=x.zeros_like();
    r.add(x,c);
    return r;
  }


  // ---- Matrix products


  template<typename TYPE>
  inline TensorView<TYPE> operator*(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    auto R=TensorView<TYPE>::zero(x.get_dims().Mprod(y.get_dims()),x.get_dev());
    if(R.asize()>0) R.add_mprod(x,y);
    return R;
  }


  // ---- Elementwise ----------------------------------------------------------------------------------------


  template<typename TYPE>
  inline TensorView<TYPE> odot(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    return x.odot(y);
  }


  // ---- Tensor product -------------------------------------------------------------------------------------


  template<typename TYPE>
  inline TensorView<TYPE> kron(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    TensorView<TYPE> R(tprod(x.get_dims(),y.get_dims()),0,x.get_dev());
    R.add_tprod(x,y);
    return R;
  }


  template<typename TYPE>
  inline TensorView<TYPE> oplus(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    CNINE_ASSRT(x.ndims()==y.ndims());
    TensorView<TYPE> R=TensorView<TYPE>::zero(x.get_dims()+y.get_dims(),x.dev);
    R.block(x.get_dims())=x;
    R.block(y.get_dims(),Gindex(x.get_dims()))=y;
    return R;
  }
  

  // ---- Scalar operations ----------------------------------------------------------------------------------

  
  template<typename TYPE>
  inline TYPE max(const TensorView<TYPE>& x){
    return x.max();
  }

  template<typename TYPE>
  inline TYPE max_abs(const TensorView<TYPE>& x){
    return x.max_abs();
  }

  template<typename TYPE>
  inline TYPE min(const TensorView<TYPE>& x){
    return x.max();
  }

  template<typename TYPE>
  inline TYPE sum(const TensorView<TYPE>& x){
    return x.sum();
  }

  template<typename TYPE>
  inline TYPE inp(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    return x.inp(y);
  }

  template<typename TYPE>
  inline TYPE norm2(const TensorView<TYPE>& x){
    return x.norm2();
  }

  template<typename TYPE>
  inline TYPE diff2(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    return x.diff2(y);
  }


  // ---- Einsum ---------------------------------------------------------------------------------------------


  template<typename TYPE>
  inline TensorView<TYPE> einsum(const string str, const TensorView<TYPE>& x, const vector<int>& rdims={}){
    Einsum1 esum(str);
    return esum(x,rdims);
  }

  template<typename TYPE>
  void einsum_add_back(const string str, const TensorView<TYPE>& x, const TensorView<TYPE>& r){
    Einsum1 esum(str);
    esum.add_einsum_back(x,r);
  }

  template<typename TYPE>
  inline TensorView<TYPE> einsum(const string str, const TensorView<TYPE>& x, const GatherMapB& gmap, const vector<int>& rdims={}){
    Einsum1 esum(str);
    return esum(x,gmap,rdims);
  }

  template<typename TYPE>
  void einsum_add_back(const string str, const TensorView<TYPE>& x, const TensorView<TYPE>& r, const GatherMapB& gmap){
    Einsum1 esum(str);
    esum.add_einsum_back(x,r,gmap);
  }


  template<typename TYPE>
  inline TensorView<TYPE> einsum(const string str, const TensorView<TYPE>& x, const TensorView<TYPE>& y, vector<int> rdims={}){
    Einsum2 esum(str);
    return esum(x,y,rdims);
  }

  template<typename TYPE>
  void einsum_add_back0(const string str, const TensorView<TYPE>& x, const TensorView<TYPE>& y, 
    const TensorView<TYPE>& r){
    Einsum2 esum(str);
    esum.add_einsum_back0(x,r,y);
  }

  template<typename TYPE>
  void einsum_add_back1(const string str, const TensorView<TYPE>& x, const TensorView<TYPE>& y, 
    const TensorView<TYPE>& r){
    Einsum2 esum(str);
    esum.add_einsum_back1(y,r,x);
  }


  // ---------------------------------------------------------------------------------------------------------


  template<typename TYPE>
  array_pool<TYPE> to_array_pool(const Tensor<TYPE>& M){ // why?
    CNINE_ASSRT(M.ndims()==2);
    CNINE_ASSRT(M.is_regular());
    array_pool<TYPE> R(M.dim(0),M.dim(1),M.get_dev());
    if(M.get_dev()==0){
      std::copy(M.mem(),M.mem()+R.get_memsize(),R.get_arr());
    }
    if(M.get_dev()==1){
      CUDA_SAFE(cudaMemcpy(R.get_arrg(),M.mem(),R.get_memsize()*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
    }
  }

  template<typename TYPE>
  array_pool<TYPE> to_array_pool(const TensorView<TYPE>& M){
    CNINE_ASSRT(M.ndims()==2);
    CNINE_ASSRT(M.is_regular());
    array_pool<TYPE> R(M.dim(0),M.dim(1),M.get_dev());
    if(M.get_dev()==0){
      std::copy(M.mem(),M.mem()+R.get_memsize(),R.get_arr());
    }
    if(M.get_dev()==1){
      CUDA_SAFE(cudaMemcpy(R.get_arrg(),M.mem(),R.get_memsize()*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
    }
    return R;
  }


  // -------------------------------------------------------------------------------------------------------
  // ---- Legacy Constructors 


  template<typename TYPE>
  inline TensorView<TYPE> Identity(const int n, const int _dev=0){
    return TensorView<TYPE>({n,n},fill_identity(), _dev);
  }

  template<typename TYPE>
  inline TensorView<TYPE> UnitVec(const int n, const int i, const int _dev=0){
    TensorView<TYPE> R({n},fill_zero(),_dev);
    R.set(i,1);
    return R;
  }




}


#endif 
  /*
  template<typename TYPE>
  TYPE inp(const TensorView<TYPE>& x, const TensorView<TYPE>& y){
    return x.inp(y);
  }

  template<typename TYPE>
  TYPE norm2(const TensorView<TYPE>& x){
    return x.norm2();
  }

  template<typename TYPE>
  TYPE norm(const TensorView<TYPE>& x){
    return x.norm();
  }

  template<typename TYPE>
  inline TensorView<TYPE> operator*(const TYPE c, const TensorView<TYPE>& x){
    TensorView<TYPE> R=x.zeros_like(); //TensorView<TYPE>::zeros_like(x);
    R.add(x,c);
    return R;
  }
  */
