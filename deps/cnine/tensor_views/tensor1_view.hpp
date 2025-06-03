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


#ifndef _Cninetensor1_view
#define _Cninetensor1_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"


#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE>
  class tensor1_view;

  #ifdef _WITH_CUDA
  //extern float Rtensor_get_cu(const float* p);
  //extern void Rtensor_set_cu(float* p, const float v);
  //extern void Rtensor_inc_cu(float* p, const float v);
  //extern void Rtensor_copy_cu(const tensor1_view& r, const tensor1_view& x, const cudaStream_t& stream);
  //extern void Rtensor_add_cu(const tensor1_view& r, const tensor1_view& x, const cudaStream_t& stream);
  #endif 


  template<typename TYPE>
  class tensor1_view{
  public:

    TYPE* arr;
    int n0;
    int s0;
    int dev=0;

  public:

    tensor1_view(){}

    tensor1_view(TYPE* _arr): 
      arr(_arr){}

    tensor1_view(TYPE* _arr, const int _n0, const int _s0, const int _dev=0): 
      arr(_arr), n0(_n0), s0(_s0), dev(_dev){}

    tensor1_view(TYPE* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dev(_dev){
      assert(_dims.size()==1);
      n0=_dims[0];
      s0=_strides[0];
    }


  public: // ---- Assignment --------------------------------------------------------------------------------


    tensor1_view& operator=(const tensor1_view& x){
      assert(dev==0);
      assert(x.dev==0);
      assert(n0==x.n0);
      for(int i=0; i<n0; i++) 
	arr[i*s0]=x.arr[i*x.s0];
      return *this;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    virtual bool is_regular() const{
      if(s0!=1) return false;
      return true;
    }

    Gdims get_dims() const{
      return Gdims({n0});
    }

    Gstrides get_strides() const{
      return Gstrides(s0);
    }

    virtual TYPE operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::tensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      CPUCODE(return arr[s0*i0]);
      GPUCODE(return Rtensor_get_cu(arr+s0*i0));
      return 0;
    }

    void set(const int i0, TYPE x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::tensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      CPUCODE(arr[s0*i0]=x);
      GPUCODE(Rtensor_set_cu(arr+s0*i0,x));
    }

    void inc(const int i0, TYPE x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::tensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      arr[s0*i0]+=x;
      GPUCODE(Rtensor_inc_cu(arr+s0*i0,x));
    }

    bool operator<(const tensor1_view& y) const{
      CNINE_ASSRT(n0==y.n0);
      for(int i=0; i<n0; i++){
	if(*(arr+i*s0)<*(y.arr+i*y.s0)) return true;
	if(*(arr+i*s0)>*(y.arr+i*y.s0)) return false;
      }
      return false;
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    TYPE sum() const{
      CNINE_CPUONLY();
      TYPE t=0;
      for(int i=0; i<n0; i++)
	t+=arr[i*s0];
      return t;
    }

    tensor1_view block(const int i0, const int m0) const{
      return tensor1_view(arr+i0*s0,m0,s0,dev);
    }

    void set(const TYPE v){
      CNINE_CPUONLY();
      for(int i=0; i<n0; i++)
	arr[i*s0]=v;
    }

    void set(const tensor1_view& x) const{
      CNINE_DEVICE_SAME(x);
      assert(x.n0==n0);
      if(is_regular() && x.is_regular()){
	CPUCODE(std::copy(x.arr,x.arr+n0,arr));
	GPUCODE(CUDA_SAFE(cudaMemcpy(arr,x.arr,n0*sizeof(TYPE),cudaMemcpyDeviceToDevice)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) set(i0,x(i0)));
	GPUCODE(CUDA_STREAM(Rtensor_copy_cu(*this,x,stream)));
      }
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add(const TYPE v){
      CNINE_CPUONLY();
      for(int i=0; i<n0; i++)
	arr[i*s0]+=v;
    }
    
    void add(const tensor1_view& x) const{
      CNINE_DEVICE_SAME(x);
      assert(x.n0==n0);
      if(is_regular() && x.is_regular()){
	CPUCODE(stdadd<TYPE>(x.arr,x.arr+n0,arr));
	GPUCODE(const TYPE alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&alpha,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) set(i0,x(i0)));
	GPUCODE(const TYPE alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&alpha,x.arr,x.s0,arr,s0)));
      }
    }

    void operator+=(const tensor1_view& x) const{
      return add(x);
    }


  public: // ---- Reductions --------------------------------------------------------------------------------


    TYPE reduce() const{
      TYPE t=0;
      CPUCODE(for(int i=0; i<n0; i++) t+=arr[i*s0]);
      GPUCODE(CUBLAS_SAFE(cublasSasum(cnine_cublas,n0,arr,s0,&t)));
      return t;
    }


  public: // ---- Broadcasting ------------------------------------------------------------------------------


    void broadcast(const TYPE v){
      assert(is_regular());
      CNINE_CPUONLY();
      CPUCODE(std::fill_n(arr,n0,v));
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<TYPE> gtensor() const{
      Gtensor<TYPE> R({n0},fill::raw);
      for(int i0=0; i0<n0; i0++)
	R(i0)=(*this)(i0);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string repr() const{
      return "<tensor1_view"+get_dims().str()+get_strides().str()+":"+to_string(dev)+">";
    }

    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const tensor1_view& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename TYPE>
  inline tensor1_view<TYPE> repeat0(const TYPE x, const int n){
    TYPE* arr=new TYPE[1];
    *arr=x;
    return tensor1_view<TYPE>(arr,n,0,0);
  }

}


namespace std{

  template<typename TYPE>
  struct hash<cnine::tensor1_view<TYPE> >{
  public:
    size_t operator()(const cnine::tensor1_view<TYPE>& x) const{
      size_t t=hash<int>()(x.n0);
      for(int i=0; i<x.n0; i++)
	t=(t^hash<TYPE>()(*(x.arr+i*x.s0)))<<1;
      return t;
    }
  };
}


#endif 
