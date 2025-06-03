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


#ifndef _cnine_tensor3_view
#define _cnine_tensor3_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "tensor2_view.hpp"

#ifdef _WITH_CUDA
//extern void batched_add_cu(float* rarr, const float* arr, const int b, const int sb, const int n, const int s, const cudaStream_t& stream);
#endif 


namespace cnine{

  template<typename TYPE>
  class tensor3_view;

  #ifdef _WITH_CUDA
  //extern float Rtensor_get_cu(const float* p);
  //extern void Rtensor_set_cu(float* p, const float v);
  //extern void Rtensor_inc_cu(float* p, const float v);
  //extern void Rtensor_copy_cu(const tensor3_view& r, const tensor3_view& x, const cudaStream_t& stream);
  //extern void Rtensor_add_cu(const tensor3_view& r, const tensor3_view& x, const cudaStream_t& stream);
  //extern void Rtensor_add_cu(const tensor3_view& r, const tensor3_view& x, const float c, const cudaStream_t& stream);
  #endif 


  template<typename TYPE>
  class tensor3_view{
  public:

    TYPE* arr;
    int n0,n1,n2;
    int s0,s1,s2;
    int dev=0;

  public:

    tensor3_view(){}

    tensor3_view(float* _arr): 
      arr(_arr){}

    tensor3_view(float* _arr, const int _n0, const int _n1, const int _n2, 
      const int _s0, const int _s1, const int _s2, const int _dev=0): 
      arr(_arr), n0(_n0), n1(_n1), n2(_n2), s0(_s0), s1(_s1), s2(_s2), dev(_dev){}

    tensor3_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dev(_dev){
      CNINE_ASSRT(_dims.size()==3);
      n0=_dims[0];
      n1=_dims[1];
      n2=_dims[2];
      s0=_strides[0];
      s1=_strides[1];
      s2=_strides[2];
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    Gdims get_dims() const{
      return Gdims(n0,n1,n2);
    }

    Gstrides get_strides() const{
      return Gstrides(s0,s1,s2);
    }

    bool is_regular() const{
      if(s2!=1) return false;
      if(s1!=n2) return false;
      if(s0!=n1*s1) return false;
      return true;
    }

    float operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=n0 || i1>=n1 || i2>=n2) 
	  throw std::out_of_range("cnine::tensor3_view: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+Gdims({n0,n1,n2}).str()));
      CPUCODE(return arr[s0*i0+s1*i1+s2*i2]);
      //GPUCODE(return Rtensor_get_cu(arr+s0*i0+s1*i1+s2*i2));
      return 0;
    }

    virtual void set(const int i0, const int i1, const int i2, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=n0 || i1>=n1 || i2>=n2) 
	  throw std::out_of_range("cnine::tensor3_view: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+Gdims({n0,n1,n2}).str()));
      CPUCODE(arr[s0*i0+s1*i1+s2*i2]=x);
      //GPUCODE(Rtensor_set_cu(arr+s0*i0+s1*i1+s2*i2,x));
    }

    virtual void inc(const int i0, const int i1, const int i2, float x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=n0 || i1>=n1 || i2>=n2) 
	  throw std::out_of_range("cnine::tensor3_view: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+Gdims({n0,n1,n2}).str()));
      CPUCODE(arr[s0*i0+s1*i1+s2*i2]+=x);
      //GPUCODE(Rtensor_inc_cu(arr+s0*i0+s1*i1+s2*i2,x));
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    void set(const float v){
      CNINE_CPUONLY();
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    arr[i0*s0+i1*s1+i2*s2]=v;
    }

    void set(const tensor3_view& x) const{
      CNINE_DEVICE_SAME(x);
      if(x.n0==n0 && x.n1==n1 && x.n2==n2 && is_regular() && x.is_regular()){
	CPUCODE(std::copy(x.arr,x.arr+n0*n1*n2,arr));
	//GPUCODE(CUDA_SAFE(cudaMemcpy(arr,x.arr,n0*n1*sizeof(float),cudaMemcpyDeviceToDevice)));
      }else{
	CPUCODE(for(int i0=0; i0<x.n0; i0++) for(int i1=0; i1<x.n1; i1++) for(int i2=0; i2<x.n2; i2++) set(i0,i1,i2,x(i0,i1,i2));});
      //GPUCODE(CUDA_STREAM(Rtensor_copy_cu(*this,x,stream)));
      }
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add(const tensor3_view& x) const{
      CNINE_DEVICE_SAME(x);
      if(x.n0==n0 && x.n1==n1 && x.n2==n2 && is_regular() && x.is_regular()){
	CPUCODE(stdadd<float>(x.arr,x.arr+n0*n1*n2,arr));
	//GPUCODE(const float alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0*n1*n2,&alpha,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<x.n0; i0++) for(int i1=0; i1<x.n1; i1++) for(int i2=0; i2<x.n2; i2++) {inc(i0,i1,i2,x(i0,i1,i2));});
	//GPUCODE(CUDA_STREAM(Rtensor_add_cu(*this,x,stream)));
      }
    }

    void add(const tensor3_view& x, const float c) const{
      CNINE_DEVICE_SAME(x);
      if(x.n0==n0 && x.n1==n1 && x.n2==n2 && is_regular() && x.is_regular()){
	CPUCODE(stdadd<float>(x.arr,x.arr+n0*n1*n2,arr));
	//GPUCODE(CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0*n1*n2,&c,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<x.n0; i0++) for(int i1=0; i1<x.n1; i1++) for(int i2=0; i2<x.n2; i2++) {inc(i0,i1,i2,x(i0,i1,i2)*c);});
	//GPUCODE(CUDA_STREAM(Rtensor_add_cu(*this,x,c,stream)));
      }
    }

    void operator+=(const tensor3_view& x) const{
      return add(x);
    }

    // batchwise 
    void add_mprod(const tensor2_view<TYPE>& x, const tensor3_view& y){
      CNINE_ASSRT(n0==y.n0);
      CNINE_ASSRT(n1==x.n0);
      CNINE_ASSRT(n2==y.n2);
      CNINE_ASSRT(x.n1==y.n1);
      CNINE_ASSRT(dev==x.dev);
      CNINE_ASSRT(dev==y.dev);
      
      if(dev==0){
	for(int b=0; b<n0; b++)
	  slice0(b).add_mprod(x,y.slice0(b));
      }
      if(dev==1){
	float alpha=1.0;
	CUBLAS_SAFE(cublasSgemmStridedBatched(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,n2,n1,x.n1,
	    &alpha,y.arr,y.s1,y.s0,
	    x.arr,x.s0,0,
	    &alpha,arr,s1,s0,n0));
      }
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    void sum0_into(const tensor2_view<TYPE>& r){
      CNINE_CPUONLY();
      assert(r.n0==n1);
      assert(r.n1==n2);
      for(int i1=0; i1<n1; i1++)
	for(int i2=0; i2<n2; i2++){
	  float t=0; 
	  for(int i0=0; i0<n0; i0++) 
	    t+=arr[s0*i0+s1*i1+s2*i2];
	  r.inc(i1,i2,t);
	}
    }

    void avg0_into(const tensor2_view<TYPE>& r){
      CNINE_CPUONLY();
      assert(r.n0==n1);
      assert(r.n1==n2);
      for(int i1=0; i1<n1; i1++)
	for(int i2=0; i2<n2; i2++){
	  float t=0; 
	  for(int i0=0; i0<n0; i0++) 
	    t+=arr[s0*i0+s1*i1+s2*i2];
	  r.inc(i1,i2,t/n0);
	}
    }

    void sum1_into(const tensor2_view<TYPE>& r){
      CNINE_CPUONLY();
      assert(r.n0==n0);
      assert(r.n1==n2);
      for(int i0=0; i0<n0; i0++) 
	for(int i2=0; i2<n2; i2++){
	  float t=0; 
	  for(int i1=0; i1<n1; i1++)
	    t+=arr[s0*i0+s1*i1+s2*i2];
	  r.inc(i0,i2,t);
	}
    }

    void avg1_into(const tensor2_view<TYPE>& r){
      CNINE_CPUONLY();
      assert(r.n0==n0);
      assert(r.n1==n2);
      for(int i0=0; i0<n0; i0++) 
	for(int i2=0; i2<n2; i2++){
	  float t=0; 
	  for(int i1=0; i1<n1; i1++)
	    t+=arr[s0*i0+s1*i1+s2*i2];
	  r.inc(i0,i2,t/n1);
	}
    }

    void sum2_into(const tensor2_view<TYPE>& r){
      CNINE_CPUONLY();
      assert(r.n0==n0);
      assert(r.n1==n1);
      for(int i0=0; i0<n0; i0++) 
	for(int i1=0; i1<n1; i1++){
	  float t=0; 
	  for(int i2=0; i2<n2; i2++)
	    t+=arr[s0*i0+s1*i1+s2*i2];
	  r.inc(i0,i1,t);
	}
    }

    void avg2_into(const tensor2_view<TYPE>& r){
      CNINE_CPUONLY();
      assert(r.n0==n0);
      assert(r.n1==n1);
      for(int i0=0; i0<n0; i0++) 
	for(int i1=0; i1<n1; i1++){
	  float t=0; 
	  for(int i2=0; i2<n2; i2++)
	    t+=arr[s0*i0+s1*i1+s2*i2];
	  r.inc(i0,i1,t/n2);
	}
    }

    void sum01_into(const Rtensor1_view& r){
      CNINE_CPUONLY();
      assert(r.n0==n2);
	for(int i2=0; i2<n2; i2++){
	  float t=0; 
	  for(int i0=0; i0<n0; i0++) 
	    for(int i1=0; i1<n1; i1++)
	      t+=arr[s0*i0+s1*i1+s2*i2];
	  r.inc(i2,t);
	}
    }

    void avg01_into(const Rtensor1_view& r){
      CNINE_CPUONLY();
      assert(r.n0==n2);
	for(int i2=0; i2<n2; i2++){
	  float t=0; 
	  for(int i0=0; i0<n0; i0++) 
	    for(int i1=0; i1<n1; i1++)
	      t+=arr[s0*i0+s1*i1+s2*i2];
	  r.inc(i2,t/n0/n1);
	}
    }


  public: // ---- Reductions --------------------------------------------------------------------------------


    void reduce0_destructively_into(const tensor2_view<TYPE>& r) const{
      reduce0_destructively();
      r.add(slice0(0));
    }

    void reduce1_destructively_into(const tensor2_view<TYPE>& r) const{
      reduce1_destructively();
      r.add(slice1(0));
    }

    void reduce2_destructively_into(const tensor2_view<TYPE>& r) const{
      reduce2_destructively();
      r.add(slice2(0));
    }


    void reduce0_destructively() const{
      fuse12().reduce0_destructively();
    }

    void reduce1_destructively() const{

      if(dev==0){
	if(is_regular()){
	  for(int i0=0; i0<n0; i0++)
	    slice0(i0).reduce0_destructively();
	}else{
	  for(int i0=0; i0<n0; i0++)
	    for(int i2=0; i2<n2; i2++){
	      float t=0;
	      for(int i1=0; i1<n1; i1++)
		t+=arr[s0*i0+s1*i1+s2*i2];
	      arr[s0*i0+s2*i2]=t;
	    }
	}
      }
      
      if(dev==1){
	if(is_regular()){
	  int a=1; while(a<n1) a*=2; a/=2;
	  #ifdef _WITH_CUDA
	  cudaStream_t stream;
	  CUDA_SAFE(cudaStreamCreate(&stream));
	  batched_add_cu(arr,arr+a*s1,n0,s0,(n1-a)*s1,s2,stream); a/=2;
	  for(;a>0;a/=2)
	    batched_add_cu(arr,arr+a*s1,n0,s0,a*s1,s2,stream);
	  CUDA_SAFE(cudaStreamDestroy(stream));
	  #endif 
	}else{
	  for(int i0=0; i0<n0; i0++)
	    slice0(i0).reduce0_destructively();
	}
      }

    }

    void reduce2_destructively() const{
      fuse01().reduce1_destructively();
    }


  public: // ---- Broadcasting ------------------------------------------------------------------------------


    void broadcast0(const tensor2_view<TYPE>& x){
      assert(x.n0==n1);
      assert(x.n1==n2);
      fuse12().broadcast0(x.fuse01());
    }

    void broadcast1(const tensor2_view<TYPE>& x){
      assert(x.n0==n0);
      assert(x.n1==n2);
      for(int i0=0; i0<n0; i0++)
	slice0(i0).broadcast0(x.slice0(i0));
    }

    void broadcast2(const tensor2_view<TYPE>& x){
      assert(x.n0==n0);
      assert(x.n1==n1);
      fuse01().broadcast1(x.fuse01());
    }


  public: // ---- Other views -------------------------------------------------------------------------------


    tensor2_view<TYPE> diag01() const{
      assert(n0==n1);
      return tensor2_view<TYPE>(arr,n0,n2,s0+s1,s2,dev);
    }

    tensor2_view<TYPE> diag02() const{
      assert(n0==n2);
      return tensor2_view<TYPE>(arr,n0,n1,s0+s2,s1,dev);
    }

    tensor2_view<TYPE> diag12() const{
      assert(n1==n2);
      return tensor2_view<TYPE>(arr,n0,n1,s0,s1+s2,dev);
    }

    tensor3_view transp01() const{
      return tensor3_view(arr,n1,n0,n2,s1,s0,s2,dev);
    }

    tensor2_view<TYPE> slice0(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::tensor3_view:slice0(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]");)
      return tensor2_view<TYPE>(arr+i*s0,n1,n2,s1,s2,dev);
    }

    tensor2_view<TYPE> slice1(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) 
	  throw std::out_of_range("cnine::tensor3_view:slice1(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]");)
      return tensor2_view<TYPE>(arr+i*s1,n0,n2,s0,s2,dev);
    }

    tensor2_view<TYPE> slice2(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n2) 
	  throw std::out_of_range("cnine::tensor3_view:slice2(int): index "+to_string(i)+" out of range of [0,"+to_string(n2-1)+"]");)
      return tensor2_view<TYPE>(arr+i*s2,n0,n1,s0,s1,dev);
    }

    tensor2_view<TYPE> fuse01() const{
      assert(is_regular());
      return tensor2_view<TYPE>(arr,n0*n1,n2,s1,s2,dev);
    }    

    tensor2_view<TYPE> fuse12() const{
      assert(is_regular());
      return tensor2_view<TYPE>(arr,n0,n1*n2,s0,s2,dev);
    }    

    tensor3_view block(const int i0, const int i1, const int i2, int m0=-1, int m1=-1, int m2=-1) const{
      if(m0==-1) m0=n0-i0;
      if(m1==-1) m1=n1-i1;
      if(m2==-1) m2=n2-i2;
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=n0 || i1>=n1 || i2>=n2) 
	  throw std::out_of_range("cnine::tensor3_view::block: index "+Gindex({i0,i1,i2}).str()+" out of range of size "+Gindex({n0,n1,n2}).str()));
      CNINE_CHECK_RANGE(if(i0+m0<0 || i1+m1<0 || i2+m2<0 || i0+m0>n0 || i1+m1>n1|| i2+m2>n2) 
	  throw std::out_of_range("cnine::tensor3_view::block: end index "+Gindex({i0+m0,i1+m1,i2+m2}).str()+" out of range of size "+Gindex({n0,n1,n2}).str()));
      return tensor3_view(arr+i0*s0+i1*s1+i2*s2,m0,m1,m2,s0,s1,s2,dev);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<float> gtensor() const{
      Gtensor<float> R({n0,n1,n2},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    R(i0,i1,i2)=(*this)(i0,i1,i2);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string repr() const{
      return "<tensor3_view"+get_dims().str()+get_strides().str()+":"+to_string(dev)+">";
    }

    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const tensor3_view& x){
      stream<<x.str(); return stream;
    }


  };


  inline tensor3_view split0(const tensor2_view<TYPE>& x, const int i, const int j){
    assert(i*j==x.n0);
    return tensor3_view(x.arr,i,j,x.n1,x.s0*j,x.s0,x.s1,x.dev);
  }

  inline tensor3_view split1(const tensor2_view<TYPE>& x, const int i, const int j){
    assert(i*j==x.n1);
    return tensor3_view(x.arr,x.n0,i,j,x.s0,x.s1*j,x.s1,x.dev);
  }
 
  inline tensor3_view repeat0(const tensor2_view<TYPE>& x, const int n){
    return tensor3_view(x.arr,n,x.n0,x.n1,0,x.s0,x.s1,x.dev);
  }

  inline tensor3_view repeat1(const tensor2_view<TYPE>& x, const int n){
    return tensor3_view(x.arr,x.n0,n,x.n1,x.s0,0,x.s1,x.dev);
  }

  inline tensor3_view repeat2(const tensor2_view<TYPE>& x, const int n){
    return tensor3_view(x.arr,x.n0,x.n1,n,x.s0,x.s1,0,x.dev);
  }


}


#endif 
