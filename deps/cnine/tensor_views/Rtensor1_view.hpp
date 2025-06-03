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


#ifndef _CnineRtensor1_view
#define _CnineRtensor1_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"


#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  class Rtensor1_view;

#ifdef _WITH_CUDA
  extern void Rtensor_add_ReLU_cu(Rtensor1_view& r, const Rtensor1_view& x, const float alpha, const cudaStream_t& stream);
  extern void Rtensor_add_ReLU_back_cu(Rtensor1_view& r, const Rtensor1_view& g, const Rtensor1_view& x, const float alpha, const cudaStream_t& stream);
#endif 

  class Rtensor1_view;

  #ifdef _WITH_CUDA
  extern float Rtensor_get_cu(const float* p);
  extern void Rtensor_set_cu(float* p, const float v);
  extern void Rtensor_inc_cu(float* p, const float v);
  extern void Rtensor_copy_cu(const Rtensor1_view& r, const Rtensor1_view& x, const cudaStream_t& stream);
  extern void Rtensor_add_cu(const Rtensor1_view& r, const Rtensor1_view& x, const cudaStream_t& stream);
  #endif 


  class Rtensor1_view{
  public:

    float* arr;
    int n0;
    int s0;
    int dev=0;

  public:

    Rtensor1_view(){}

    Rtensor1_view(float* _arr): 
      arr(_arr){}

    //Rtensor1_view(float* _arr, const int _n0, const int _s0): 
    //arr(_arr), n0(_n0), s0(_s0){}

    Rtensor1_view(float* _arr, const int _n0, const int _s0, const int _dev=0): 
      arr(_arr), n0(_n0), s0(_s0), dev(_dev){}

    Rtensor1_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dev(_dev){
      CNINE_ASSRT(_dims.size()==1);
      n0=_dims[0];
      s0=_strides[0];
    }

    Rtensor1_view(float* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a):
      arr(_arr){
      CNINE_ASSRT(_strides.is_regular(_dims));
      CNINE_ASSRT(a.is_contiguous());
      CNINE_ASSRT(a.covers(_dims.size()));
      n0=_dims.unite(a);
      s0=_strides[a.back()];
    }


  public: // ---- Assignment --------------------------------------------------------------------------------


    Rtensor1_view& operator=(const Rtensor1_view& x){
      CNINE_ASSRT(dev==0);
      CNINE_ASSRT(x.dev==0);
      CNINE_ASSRT(n0==x.n0);
      for(int i=0; i<n0; i++) 
	arr[i*s0]=x(i);
	//arr[i*s0]=x.arr[i*x.s0];
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

    virtual float operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::Rtensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      CPUCODE(return arr[s0*i0]);
      GPUCODE(return Rtensor_get_cu(arr+s0*i0));
      return 0;
    }

    void set(const int i0, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::Rtensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      CPUCODE(arr[s0*i0]=x);
      GPUCODE(Rtensor_set_cu(arr+s0*i0,x));
    }

    void inc(const int i0, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::Rtensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      arr[s0*i0]+=x;
      GPUCODE(Rtensor_inc_cu(arr+s0*i0,x));
    }

    float sum() const{
      float t=0;
      for(int i=0; i<n0; i++)
	t+=arr[i*s0];
      return t;
    }

    Rtensor1_view block(const int i0, const int m0) const{
      return Rtensor1_view(arr+i0*s0,m0,s0,dev);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    void set(const float v){
      CNINE_CPUONLY();
      for(int i=0; i<n0; i++)
	arr[i*s0]=v;
    }

    void set(const Rtensor1_view& x) const{
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n0);
      if(is_regular() && x.is_regular()){
	CPUCODE(std::copy(x.arr,x.arr+n0,arr));
	GPUCODE(CUDA_SAFE(cudaMemcpy(arr,x.arr,n0*sizeof(float),cudaMemcpyDeviceToDevice)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) set(i0,x(i0)));
	GPUCODE(CUDA_STREAM(Rtensor_copy_cu(*this,x,stream)));
      }
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add(const float v){
      CNINE_CPUONLY();
      for(int i=0; i<n0; i++)
	arr[i*s0]+=v;
    }
    
    void add(const Rtensor1_view& x) const{
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n0);
      if(is_regular() && x.is_regular()){
	CPUCODE(stdadd<float>(x.arr,x.arr+n0,arr));
	GPUCODE(const float alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&alpha,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) inc(i0,x(i0)));
	GPUCODE(const float alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&alpha,x.arr,x.s0,arr,s0)));
      }
    }

    void add(const Rtensor1_view& x, const float c) const{
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n0);
      if(is_regular() && x.is_regular()){
	CPUCODE(for(int i0=0; i0<n0; i0++) arr[i0]+=c*x.arr[i0];);
	GPUCODE(CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&c,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) inc(i0,c*x(i0)));
	GPUCODE(CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&c,x.arr,x.s0,arr,s0)));
      }
    }

    void operator+=(const Rtensor1_view& x) const{
      return add(x);
    }


  public: // ---- Reductions --------------------------------------------------------------------------------


    float reduce() const{
      float t=0;
      CPUCODE(for(int i=0; i<n0; i++) t+=arr[i*s0]);
      GPUCODE(CUBLAS_SAFE(cublasSasum(cnine_cublas,n0,arr,s0,&t)));
      return t;
    }


  public: // ---- Broadcasting ------------------------------------------------------------------------------


    void broadcast(const float v){
      CNINE_ASSRT(is_regular());
      CNINE_CPUONLY();
      CPUCODE(std::fill_n(arr,n0,v));
    }


  public: // ---- ReLU ---------------------------------------------------------------------------------------


    void add_ReLU(const Rtensor1_view& x, const float alpha){
      CNINE_ASSRT(n0==x.n0);
      if(dev==0){
	float a=1.0-alpha;
	for(int i=0; i<n0; i++)
	  arr[i*s0]+=((x.arr[i*x.s0]>0)*a+alpha)*x.arr[i*s0];
      }
      if(dev==1){
	CUDA_STREAM(Rtensor_add_ReLU_cu(*this,x,alpha,stream));
      }
    }

    void add_ReLU_back(const Rtensor1_view& g, const Rtensor1_view& x, const float alpha){
      CNINE_ASSRT(n0==g.n0);
      CNINE_ASSRT(n0==x.n0);
      if(dev==0){
	float a=1.0-alpha;
	for(int i=0; i<n0; i++)
	  arr[i*s0]+=((x.arr[i*x.s0]>0)*a+alpha)*g.arr[i*g.s0];
      }
      if(dev==1){
	CUDA_STREAM(Rtensor_add_ReLU_back_cu(*this,g,x,alpha,stream));
      }
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<float> gtensor() const{
      Gtensor<float> R({n0},fill::raw);
      for(int i0=0; i0<n0; i0++)
	R(i0)=(*this)(i0);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string repr() const{
      return "<Rtensor1_view"+get_dims().str()+get_strides().str()+":"+to_string(dev)+">";
    }

    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Rtensor1_view& x){
      stream<<x.str(); return stream;
    }


  };


  inline Rtensor1_view repeat0(const float x, const int n){
    float* arr=new float[1];
    *arr=x;
    return Rtensor1_view(arr,n,0,0);
  }

}


#endif 
