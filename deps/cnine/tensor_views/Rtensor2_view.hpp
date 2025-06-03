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


#ifndef _CnineRtensor2_view
#define _CnineRtensor2_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor1_view.hpp"

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
//#include <thrust/device_vector.h>
extern cublasHandle_t cnine_cublas;
extern float* cuda_oneS;
#endif 


namespace cnine{

  class Rtensor2_view;

  #ifdef _WITH_CUDA
  extern float Rtensor_get_cu(const float* p);
  extern void Rtensor_set_cu(float* p, const float v);
  extern void Rtensor_inc_cu(float* p, const float v);
  extern void Rtensor_copy_cu(const Rtensor2_view& r, const Rtensor2_view& x, const cudaStream_t& stream);
  extern void Rtensor_add_cu(const Rtensor2_view& r, const Rtensor2_view& x, const cudaStream_t& stream);
  extern void Rtensor_add_cu(const Rtensor2_view& r, const Rtensor2_view& x, const float c, const cudaStream_t& stream);
  extern void Rtensor_sum0_into_cu(const Rtensor1_view& r, const Rtensor2_view& x, const cudaStream_t& stream);
  extern void Rtensor_sum0_into_cu(const Rtensor1_view& r, const Rtensor2_view& x, const float c, const cudaStream_t& stream);
  #endif 

  
  class Rtensor2_view{
  public:

    float* arr;
    int n0,n1;
    int s0,s1;
    int dev=0;

  public:

    Rtensor2_view(){}

    Rtensor2_view(float* _arr): 
      arr(_arr){}

    Rtensor2_view(float* _arr, const int _n0, const int _n1, const int _s0, const int _s1, const int _dev=0): 
      arr(_arr), n0(_n0), n1(_n1), s0(_s0), s1(_s1), dev(_dev){}

    Rtensor2_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dev(_dev){
      CNINE_ASSRT(_dims.size()==2);
      n0=_dims[0];
      n1=_dims[1];
      s0=_strides[0];
      s1=_strides[1];
    }

    Rtensor2_view(float* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a, const GindexSet& b, const int _dev=0):
      arr(_arr), dev(_dev){
      CNINE_ASSRT(_strides.is_regular(_dims));
      CNINE_ASSRT(a.is_contiguous());
      CNINE_ASSRT(b.is_contiguous());
      CNINE_ASSRT(a.is_disjoint(b));
      CNINE_ASSRT(a.covers(_dims.size(),b));
      n0=_dims.unite(a);
      n1=_dims.unite(b);
      s0=_strides[a.back()];
      s1=_strides[b.back()];
    }


  public: // ---- Copy --------------------------------------------------------------------------------------

    

  public: // ---- Access ------------------------------------------------------------------------------------

    
    virtual bool is_regular() const{
      if(s1!=1) return false;
      if(s0!=n1) return false;
      return true;
    }

    virtual bool is_transpose() const{
      if(s1!=n0) return false;
      if(s0!=1) return false;
      return true;
    }

    Gdims get_dims() const{
      return Gdims({n0,n1});
    }

    Gstrides get_strides() const{
      return Gstrides(s0,s1);
    }

    virtual float operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=n0 || i1>=n1) 
	  throw std::out_of_range("cnine::Rtensor2_view: index "+Gindex({i0,i1}).str()+" out of range of view size "+Gdims({n0,n1}).str()));
      CPUCODE(return arr[s0*i0+s1*i1]);
      GPUCODE(return Rtensor_get_cu(arr+s0*i0+s1*i1));
      return 0;
    }

    virtual void set(const int i0, const int i1, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=n0 || i1>=n1) 
	  throw std::out_of_range("cnine::Rtensor2_view: index "+Gindex({i0,i1}).str()+" out of range of view size "+Gdims({n0,n1}).str()));
      CPUCODE(arr[s0*i0+s1*i1]=x);
      GPUCODE(Rtensor_set_cu(arr+s0*i0+s1*i1,x));
    }

    virtual void inc(const int i0, const int i1, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=n0 || i1>=n1) 
	  throw std::out_of_range("cnine::Rtensor2_view: index "+Gindex({i0,i1}).str()+" out of range of view size "+Gdims({n0,n1}).str()));
      CPUCODE(arr[s0*i0+s1*i1]+=x);
      GPUCODE(Rtensor_inc_cu(arr+s0*i0+s1*i1,x));
    }

    Rtensor2_view block(const int i0, const int i1, int m0=-1, int m1=-1) const{
      CNINE_CPUONLY();
      if(m0<0) m0=n0-i0;
      if(m1<0) m1=n1-i1;
      return Rtensor2_view(arr+i0*s0+i1*s1,m0,m1,s0,s1,dev);
    }

    Rtensor2_view cols(const int i1, int m1=-1) const{
      CNINE_CPUONLY();
      if(m1<0) m1=n1-i1;
      return Rtensor2_view(arr+i1*s1,n0,m1,s0,s1,dev);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    void set(const float v){
      CNINE_CPUONLY();
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  arr[i0*s0+i1*s1]=v;
    }

    void set(const Rtensor2_view& x) const{
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n1);
      if(is_regular() && x.is_regular()){
	CPUCODE(std::copy(x.arr,x.arr+n0*n1,arr));
	GPUCODE(CUDA_SAFE(cudaMemcpy(arr,x.arr,n0*n1*sizeof(float),cudaMemcpyDeviceToDevice)));
      }else{
	CPUCODE(for(int i0=0; i0<x.n0; i0++) for(int i1=0; i1<x.n1; i1++) {set(i0,i1,x(i0,i1));});
	GPUCODE(CUDA_STREAM(Rtensor_copy_cu(*this,x,stream)));
      }
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add(const float v){
      CNINE_CPUONLY();
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  arr[i0*s0+i1*s1]+=v;
    }

    void add(const Rtensor2_view& x) const{
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n1);
      if(x.n0*x.n1==0) return;
      if(is_regular() && x.is_regular()){
	CPUCODE(stdadd<float>(x.arr,x.arr+n0*s0,arr));
	GPUCODE(const float alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0*n1,&alpha,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<x.n0; i0++) for(int i1=0; i1<x.n1; i1++) {inc(i0,i1,x(i0,i1));});
	GPUCODE(CUDA_STREAM(Rtensor_add_cu(*this,x,stream)));
      }
    }

    void add(const Rtensor2_view& x, const float c) const{
      CNINE_DEVICE_SAME(x);
      CNINE_CPUONLY();
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n1);
      if(x.n0*x.n1==0) return;
      if(is_regular() && x.is_regular()){
 	CPUCODE(stdadd<float>(x.arr,x.arr+n0*s0,arr,c));
	GPUCODE(CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0*n1,&c,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) for(int i1=0; i1<n1; i1++) {inc(i0,i1,c*x(i0,i1));});
	GPUCODE(CUDA_STREAM(Rtensor_add_cu(*this,x,c,stream)));
      }
    }

    void operator+=(const Rtensor2_view& x) const{
      return add(x);
    }

    void add_mprod(const Rtensor2_view& x, const Rtensor2_view& y){
      if(is_regular()){
	if(x.is_regular() && y.is_regular()){
	  add_matmul_AA(x,y);
	  return;
	}
	if(x.is_regular() && y.is_transpose()){
	  add_matmul_AT(x,y.transp());
	  return;
	}
	if(x.is_transpose() && y.is_regular()){
	  add_matmul_TA(x.transp(),y);
	  return;
	}
      }
      CNINE_UNIMPL();
    }

    void add_matmul_AA(const Rtensor2_view& x, const Rtensor2_view& y){
      const int I=x.n1;
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(y.n1==n1);
      CNINE_ASSRT(y.n0==I);
      CNINE_ASSRT(x.dev==dev);
      CNINE_ASSRT(y.dev==dev);

      if(dev==0){
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++){
	    float t=0;
	    for(int i=0; i<I; i++)
	      t+=x(a,i)*y(i,b);
	    inc(a,b,t);
	  }
      }
      if(dev==1){
	CNINE_ASSRT(s1==1);
	CNINE_ASSRT(x.s1==1);
	CNINE_ASSRT(y.s1==1);
	GPUCODE(const float alpha=1.0;
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,n1,n0,I,&alpha,
	    y.arr,y.s0,x.arr,x.s0,&alpha,arr,s0));
		)
      }
    }
    
    void add_matmul_AT(const Rtensor2_view& x, const Rtensor2_view& y){
      const int I=x.n1;
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(y.n0==n1);
      CNINE_ASSRT(y.n1==I);
      CNINE_ASSRT(x.dev==dev);
      CNINE_ASSRT(y.dev==dev);

      if(dev==0){
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++){
	    float t=0;
	    for(int i=0; i<I; i++)
	      t+=x(a,i)*y(b,i);
	    inc(a,b,t);
	  }
      }

      if(dev==1){
	CNINE_ASSRT(s1==1);
	CNINE_ASSRT(x.s1==1);
	CNINE_ASSRT(y.s1==1);
	GPUCODE(
		const float alpha=1.0;
		CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,n1,n0,I,&alpha,
		    y.arr,y.s0,x.arr,x.s0,&alpha,arr,s0));)
      }
    }
    
   void add_matmul_TA(const Rtensor2_view& x, const Rtensor2_view& y){
      const int I=x.n0;
      CNINE_ASSRT(x.n1==n0);
      CNINE_ASSRT(y.n0==I);
      CNINE_ASSRT(y.n1==n1);
      CNINE_ASSRT(x.dev==dev);
      CNINE_ASSRT(y.dev==dev);

      if(dev==0){
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++){
	    float t=0;
	    for(int i=0; i<I; i++)
	      t+=x(i,a)*y(i,b);
	    inc(a,b,t);
	  }
      }

      if(dev==1){
	CNINE_ASSRT(s1==1);
	CNINE_ASSRT(x.s1==1);
	CNINE_ASSRT(y.s1==1);
	GPUCODE(
		const float alpha=1.0;
		CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,n1,n0,y.n0,&alpha,
		    y.arr,y.s0,x.arr,x.s0,&alpha,arr,s0));)
      }
    }


    void add_mprod_to(const Rtensor1_view& y, const Rtensor1_view& x) const{
      CNINE_CPUONLY();
      CNINE_ASSRT(x.n0==n1);
      CNINE_ASSRT(y.n0==n0);
      for(int a=0; a<n0; a++){
	float t=0;
	for(int b=0; b<n1; b++)
	  t+=(*this)(a,b)*x(b);
	y.inc(a,t);
      }
    }


    void add_outer(const Rtensor1_view& x, const Rtensor1_view& y) const{
      CNINE_CPUONLY();
      CNINE_ASSRT(n0==x.n0);
      CNINE_ASSRT(n1==y.n0);
      for(int i0=0; i0<x.n0; i0++)
	for(int i1=0; i1<y.n0; i1++)
	  inc(i0,i1,x(i0)*y(i1));
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    float sum() const{
      CNINE_CPUONLY();
      if(is_regular()){
	float t=0;
	for(int i=0; i<n0*n1; i++) 
	  t+=arr[i];
	return t;
      }
      float t=0;
      for(int i=0; i<n0; i++) 
	for(int j=0; j<n1; j++) 
	  t+=(*this)(i,j);;
      return t;
    }
    
    /*
    void sum0_into(const Rtensor1_view& r){
      CNINE_CPUONLY();
      CNINE_ASSRT(r.n0==n1);
      for(int i=0; i<n1; i++){
	float t=0; for(int j=0; j<n0; j++) t+=arr[s0*j+s1*i];
	r.inc(i,t);
      }
    }
    */

    void sum1_into(const Rtensor1_view& r){
      CNINE_CPUONLY();
      CNINE_ASSRT(r.n0==n0);
      for(int i=0; i<n0; i++){
	float t=0; for(int j=0; j<n1; j++) t+=arr[s0*i+s1*j];
	r.inc(i,t);
      }
    }


  public: // ---- Reductions --------------------------------------------------------------------------------


    void sum0_into(const Rtensor1_view& r) const{
      CNINE_ASSRT(r.n0==n1);
      if(dev==0){
	for(int j=0; j<n1; j++){
	  float t=0; 
	  for(int i=0; i<n0; i++)
	    t+=arr[i*s0+j*s1];
	  r.inc(j,t);
	}
      }
      if(dev==1){
	CUDA_STREAM(Rtensor_sum0_into_cu(r,*this,stream));
      }
    }

    void avg0_into(const Rtensor1_view& r) const{
      CNINE_ASSRT(r.n0==n1);
      if(dev==0){
	for(int j=0; j<n1; j++){
	  float t=0; 
	  for(int i=0; i<n0; i++)
	    t+=arr[i*s0+j*s1];
	  r.inc(j,t/n0);
	}
      }
      if(dev==1){
	CUDA_STREAM(Rtensor_sum0_into_cu(r,*this,1.0/n0,stream));
      }
    }

    void sum1_into(const Rtensor1_view& r) const{
      CNINE_CPUONLY();
      CNINE_ASSRT(r.n0==n1);
      for(int i=0; i<n0; i++){
	float t=0; 
	for(int j=0; j<n1; j++)
	  t+=arr[i*s0+j*s1];
	r.inc(i,t);
      }
    }

    void reduce0_destructively_into(const Rtensor1_view& r) const{
      CNINE_ASSRT(r.n0==n1);
      reduce0_destructively();
      r.add(slice0(0));
    }

    void reduce1_destructively_into(const Rtensor1_view& r) const{
      CNINE_ASSRT(r.n0==n0);
      reduce1_destructively();
      r.add(slice1(0));
    }

    void reduce0_destructively() const{
      if(dev==0){
	if(is_regular()){
	  int a=1; while(a<n0) a*=2; a/=2;
	  stdadd<float>(arr+a*s0,arr+n0*s0,arr); a/=2;
	  for(;a>0;a/=2)
	    stdadd<float>(arr+a*s0,arr+2*a*s0,arr);
	}else{
	  for(int i1=0; i1<n1; i1++){
	    float t=0;
	    for(int i0=0; i0<n0; i0++) 
	      t+=(*this)(i0,i1);
	    set(0,i1,t);
	  }
	}
      }
      if(dev==1){
	if(is_regular()){
	  const float alpha=1; 
	  int a=1; while(a<n0) a*=2; a/=2;
	  CUBLAS_SAFE(cublasSaxpy(cnine_cublas,(n0-a)*s0,&alpha,arr+a*s0,1,arr,1)); a/=2;
	  for(;a>0;a/=2)
	    CUBLAS_SAFE(cublasSaxpy(cnine_cublas,a*s0,&alpha,arr+a*s0,1,arr,1));
	}else{
	  CNINE_UNIMPL();
	  //CUDA_STREAM(Rtensor2_reduce0(*this,stream));
	}
      }
    }

    void reduce1_destructively() const{
      transp().reduce0_destructively();
    }


  public: // ---- Broadcasting ------------------------------------------------------------------------------


    void broadcast0(const Rtensor1_view& x){
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n1);
      if(is_regular() && x.is_regular()){
	if(dev==0){
	  std::copy(x.arr,x.arr+n1,arr);
	  int a=1; 
	  for(; a<=n0/2; a*=2){ // this is correct
	    std::copy(arr,arr+a*n1,arr+a*n1);
	  }
	  if(n0>a) std::copy(arr,arr+(n0-a)*n1,arr+a*n1);
	}
	if(dev==1){
	  CUDA_SAFE(cudaMemcpy(arr,x.arr,n1*sizeof(float),cudaMemcpyDeviceToDevice));
	  int a=1; 
	  for(; a<=n0/2; a*=2)
	    CUDA_SAFE(cudaMemcpy(arr+a*n1,arr,a*n1*sizeof(float),cudaMemcpyDeviceToDevice));
	  if(n0>a) CUDA_SAFE(cudaMemcpy(arr+a*n1,arr,(n0-a)*n1*sizeof(float),cudaMemcpyDeviceToDevice));
	}
      }else{
	CPUCODE(for(int i1=0; i1<n1; i1++){float t=x(i1); for(int i0=0; i0<n0; i0++) set(i0,i1,t);});
	GPUCODE(CNINE_UNIMPL());
      }
    }

    void broadcast1(const Rtensor1_view& x){
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n0);
      if(dev==0){
	for(int i0=0; i0<n0; i0++){
	  float t=x(i0);
	  for(int i1=0; i1<n1; i1++)
	    set(i0,i1,t);
	}
      }
      if(dev==1){
	CNINE_UNIMPL();
	if(s0==n1*s1){ // this is wrong
	  CUBLAS_SAFE(cublasScopy(cnine_cublas,n0,x.arr,x.s0,arr,s0));
	  int a=1; 
	  for(; a<=n1/2; a*=2)
	    CUBLAS_SAFE(cublasScopy(cnine_cublas,n0*a,arr,s0,arr+a*s1,s0));
	  CUBLAS_SAFE(cublasScopy(cnine_cublas,n0*(n1-a),arr,s0,arr+a*s1,s0));
	}else{
	}
      }
    }


    void add_broadcast0(const Rtensor1_view& x, const float alpha=1.0){
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n1);
      if(x.s0==1 && s1==1){
	if(dev==0){
	  for(int i=0; i<n0; i++)
	    stdadd(x.arr,x.arr+n1,arr+i*s0);
	}
	if(dev==1){
	  float beta=1.0;
	  CUBLAS_SAFE(cublasSgemmStridedBatched(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,n1,1,1,
	      &alpha,x.arr,x.n0,0,
	      cuda_oneS,1,0,
	      &beta,arr,x.n0,s0,n0));
	}
      }else{
	CPUCODE(for(int i1=0; i1<n1; i1++){float t=x(i1); for(int i0=0; i0<n0; i0++) inc(i0,i1,t);});
	GPUCODE(CNINE_UNIMPL());
      }
    }


  public: // ---- Other views -------------------------------------------------------------------------------


    Rtensor1_view diag() const{
      CNINE_ASSRT(n0==n1);
      return Rtensor1_view(arr,n0,s0+s1,dev);
    }

    Rtensor2_view transp() const{
      return Rtensor2_view(arr,n1,n0,s1,s0,dev);
    }

    Rtensor1_view slice0(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) throw std::out_of_range("cnine::Rtensor2_view:slice0(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]"););
      return Rtensor1_view(arr+i*s0,n1,s1,dev);
    }

    Rtensor1_view slice1(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) throw std::out_of_range("cnine::Rtensor2_view:slice1(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]"););
      return Rtensor1_view(arr+i*s1,n0,s0,dev);
    }

    Rtensor1_view fuse01() const{
      return Rtensor1_view(arr,n0*n1,s1,dev);
    }


    Rtensor2_view block(const int i0, const int i1, int m0=-1, int m1=-1){
      if(m0==-1) m0=n0-i0;
      if(m1==-1) m1=n1-i1;
      if(n0==0) cout<<"Panic!"<<endl;
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=n0 || i1>=n1) 
	  throw std::out_of_range("cnine::Rtensor2_view::block: index "+Gindex({i0,i1}).str()+" out of range of size "+Gindex({n0,n1}).str()));
      CNINE_CHECK_RANGE(if(i0+m0<0 || i1+m1<0 || i0+m0>n0 || i1+m1>n1) 
	  throw std::out_of_range("cnine::Rtensor2_view::block: end index "+Gindex({i0+m0,i1+m1}).str()+" out of range of size "+Gindex({n0,n1}).str()));
      return Rtensor2_view(arr+i0*s0+i1*s1,m0,m1,s0,s1,dev);
    }

    Rtensor2_view rows(const int offs, const int n){
      return Rtensor2_view(arr+offs*s0,n,n1,s0,s1,dev);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<float> gtensor() const{
      Gtensor<float> R({n0,n1},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++){
	  R(i0,i1)=(*this)(i0,i1);
	}
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string repr() const{
      return "<Rtensor2_view"+get_dims().str()+get_strides().str()+":"+to_string(dev)+">";
    }

    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Rtensor2_view& x){
      stream<<x.str(); return stream;
    }

  };


  inline Rtensor2_view split0(const Rtensor1_view& x, const int i, const int j){
    CNINE_ASSRT(i*j==x.n0);
    return Rtensor2_view(x.arr,i,j,x.s0*j,x.s0,x.dev);
  }

  inline Rtensor2_view repeat0(const Rtensor1_view& x, const int n){
    return Rtensor2_view(x.arr,n,x.n0,0,x.s0,x.dev);
  }

  inline Rtensor2_view repeat1(const Rtensor1_view& x, const int n){
    return Rtensor2_view(x.arr,x.n0,n,x.s0,0,x.dev);
  }


  inline void add_matmul_Ax_to(const Rtensor1_view& r, const Rtensor2_view& A, const Rtensor1_view& x){
    int dev=r.dev;
    CNINE_CPUONLY();
    CNINE_ASSRT(A.n0==r.n0);
    CNINE_ASSRT(A.n1==x.n0);
    for(int i0=0; i0<r.n0; i0++){
      float t=0;
      for(int i1=0; i1<x.n0; i1++)
	t+=A(i0,i1)*x(i1);
      r.inc(i0,t);
    }
  }



}


#endif 
