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


#ifndef _CnineItensor2_view
#define _CnineItensor2_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Itensor1_view.hpp"


namespace cnine{

  
  class Itensor2_view{
  public:

    int* arr;
    int n0,n1;
    int s0,s1;
    int dev=0;

  public:

    Itensor2_view(){}

    Itensor2_view(int* _arr): 
      arr(_arr){}

    Itensor2_view(int* _arr, const int _n0, const int _n1, const int _s0, const int _s1, const int _dev=0): 
      arr(_arr), n0(_n0), n1(_n1), s0(_s0), s1(_s1), dev(_dev){}

    Itensor2_view(int* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dev(_dev){
      assert(_dims.size()==2);
      n0=_dims[0];
      n1=_dims[1];
      s0=_strides[0];
      s1=_strides[1];
    }

    Itensor2_view(int* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a, const GindexSet& b, const int _dev=0):
      arr(_arr), dev(_dev){
      assert(_strides.is_regular(_dims));
      assert(a.is_contiguous());
      assert(b.is_contiguous());
      assert(a.is_disjoint(b));
      assert(a.covers(_dims.size(),b));
      n0=_dims.unite(a);
      n1=_dims.unite(b);
      s0=_strides[a.back()];
      s1=_strides[b.back()];
    }


  public: // ---- Copy --------------------------------------------------------------------------------------

    

  public: // ---- Access ------------------------------------------------------------------------------------

    
    bool is_regular() const{
      if(s1!=1) return false;
      if(s0!=n1) return false;
      return true;
    }

    int operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=n0 || i1>=n1) 
	  throw std::out_of_range("cnine::Itensor2_view: index "+Gindex({i0,i1}).str()+" out of range of view size "+Gdims({n0,n1}).str()));
      CPUCODE(return arr[s0*i0+s1*i1]);
      //GPUCODE(return Itensor_get_cu(arr+s0*i0+s1*i1));
      return 0;
    }

    void set(const int i0, const int i1, int x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=n0 || i1>=n1) 
	  throw std::out_of_range("cnine::Itensor2_view: index "+Gindex({i0,i1}).str()+" out of range of view size "+Gdims({n0,n1}).str()));
      CPUCODE(arr[s0*i0+s1*i1]=x);
      //GPUCODE(Itensor_set_cu(arr+s0*i0+s1*i1,x));
    }

    void inc(const int i0, const int i1, int x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=n0 || i1>=n1) 
	  throw std::out_of_range("cnine::Itensor2_view: index "+Gindex({i0,i1}).str()+" out of range of view size "+Gdims({n0,n1}).str()));
      CPUCODE(arr[s0*i0+s1*i1]+=x);
      //GPUCODE(Itensor_inc_cu(arr+s0*i0+s1*i1,x));
    }

    Itensor2_view block(const int i0, const int i1, int m0=-1, int m1=-1) const{
      CNINE_CPUONLY();
      if(m0<0) m0=n0-i0;
      if(m1<0) m1=n1-i1;
      return Itensor2_view(arr+i0*s0+i1*s1,m0,m1,s0,s1,dev);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void set(const Itensor2_view& x) const{
      CNINE_DEVICE_SAME(x);
      assert(x.n0==n0);
      assert(x.n1==n1);
      if(is_regular() && x.is_regular()){
	CPUCODE(std::copy(x.arr,x.arr+n0*n1,arr));
	GPUCODE(CUDA_SAFE(cudaMemcpy(arr,x.arr,n0*n1*sizeof(int),cudaMemcpyDeviceToDevice)));
      }else{
	CPUCODE(for(int i0=0; i0<x.n0; i0++) for(int i1=0; i1<x.n1; i1++) {set(i0,i1,x(i0,i1));});
	//GPUCODE(CUDA_STREAM(Itensor_copy_cu(*this,x,stream)));
      }
    }


    void add(const Itensor2_view& x) const{
      CNINE_DEVICE_SAME(x);
      assert(x.n0==n0);
      assert(x.n1==n1);
      if(is_regular() && x.is_regular()){
	CPUCODE(stdadd<int>(x.arr,x.arr+n0*s0,arr));
	//GPUCODE(const int alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0*n1,&alpha,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<x.n0; i0++) for(int i1=0; i1<x.n1; i1++) {inc(i0,i1,x(i0,i1));});
	//GPUCODE(CUDA_STREAM(Itensor_add_cu(*this,x,stream)));
      }
    }


    void add(const Itensor2_view& x, const int c){
      CNINE_DEVICE_SAME(x);
      CNINE_CPUONLY();
      assert(x.n0==n0);
      assert(x.n1==n1);
      if(is_regular() && x.is_regular()){
 	CPUCODE(stdadd<int>(x.arr,x.arr+n0*s0,arr,c));
	//GPUCODE(CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0*n1,&c,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) for(int i1=0; i1<n1; i1++) {inc(i0,i1,c*x(i0,i1));});
	//GPUCODE(CUDA_STREAM(Itensor_add_cu(*this,x,c,stream)));
      }
    }


    void add_matmul_AA(const Itensor2_view& x, const Itensor2_view& y){
      CNINE_CPUONLY();
      const int I=x.n1;
      assert(x.n0==n0);
      assert(y.n1==n1);
      assert(y.n0==I);

      for(int a=0; a<n0; a++)
	for(int b=0; b<n1; b++){
	  int t=0;
	  for(int i=0; i<I; i++)
	    t+=x(a,i)*y(i,b);
	  inc(a,b,t);
	}
    }
    

  public: // ---- Reductions --------------------------------------------------------------------------------

    /*
    void reduce0_destructively_into(const Itensor1_view& r) const{
      assert(r.n0==n1);
      reduce0_destructively();
      r.add(slice0(0));
    }

    void reduce1_destructively_into(const Itensor1_view& r) const{
      assert(r.n0==n0);
      reduce0_destructively();
      r.add(slice1(0));
    }

    void reduce0_destructively() const{
      if(dev==0){
	if(is_regular()){
	  int a=1; while(a<n0) a*=2; a/=2;
	  stdadd<int>(arr+a*s0,arr+n0*s0,arr); a/=2;
	  for(;a>0;a/=2)
	    stdadd<int>(arr+a*s0,arr+2*a*s0,arr);
	}else{
	  for(int i1=0; i1<n1; i1++){
	    int t=0;
	    for(int i0=0; i0<n0; i0++) 
	      t+=(*this)(i0,i1);
	    set(0,i1,t);
	  }
	}
      }
      if(dev==1){
	if(is_regular()){
	  const int alpha=1; 
	  int a=1; while(a<n0) a*=2; a/=2;
	  CUBLAS_SAFE(cublasSaxpy(cnine_cublas,(n0-a)*s0,&alpha,arr+a*s0,1,arr,1)); a/=2;
	  for(;a>0;a/=2)
	    CUBLAS_SAFE(cublasSaxpy(cnine_cublas,a*s0,&alpha,arr+a*s0,1,arr,1));
	}else{
	  CNINE_UNIMPL();
	  //CUDA_STREAM(Itensor2_reduce0(*this,stream));
	}
      }
    }

    void reduce1_destructively() const{
      transp().reduce0_destructively();
    }
    */

  public: // ---- Broadcasting ------------------------------------------------------------------------------

    /*
    void broadcast0(const Itensor1_view& x){
      CNINE_DEVICE_SAME(x);
      assert(x.n0==n1);
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
	  CUDA_SAFE(cudaMemcpy(arr,x.arr,n1*sizeof(int),cudaMemcpyDeviceToDevice));
	  int a=1; 
	  for(; a<=n0/2; a*=2)
	    CUDA_SAFE(cudaMemcpy(arr+a*n1,arr,a*n1*sizeof(int),cudaMemcpyDeviceToDevice));
	  if(n0>a) CUDA_SAFE(cudaMemcpy(arr+a*n1,arr,(n0-a)*n1*sizeof(int),cudaMemcpyDeviceToDevice));
	}
      }else{
	CPUCODE(for(int i1=0; i1<n1; i1++){int t=x(i1); for(int i0=0; i0<n0; i0++) set(i0,i1,t);});
	GPUCODE(CNINE_UNIMPL());
      }
    }

    void broadcast1(const Itensor1_view& x){
      CNINE_DEVICE_SAME(x);
      assert(x.n0==n0);
      if(dev==0){
	for(int i0=0; i0<n0; i0++){
	  int t=x(i0);
	  for(int i1=0; i1<n1; i1++)
	    set(i0,i1,t);
	}
      }
      if(dev==1){
	CNINE_UNIMPL();
	if(s0==n1*s1){ // this is wrong
	  CUDA_SAFE(cublasScopy(cnine_cublas,n0,x.arr,x.s0,arr,s0));
	  int a=1; 
	  for(; a<=n1/2; a*=2)
	    CUDA_SAFE(cublasScopy(cnine_cublas,n0*a,arr,s0,arr+a*s1,s0));
	  CUDA_SAFE(cublasScopy(cnine_cublas,n0*(n1-a),arr,s0,arr+a*s1,s0));
	}else{
	}
      }
    }
    */

  public: // ---- Other views -------------------------------------------------------------------------------


    Itensor1_view diag() const{
      assert(n0==n1);
      return Itensor1_view(arr,n0,s0+s1,dev);
    }

    Itensor2_view transp() const{
      return Itensor2_view(arr,n1,n0,s1,s0,dev);
    }
    Itensor1_view slice0(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) throw std::out_of_range("cnine::Itensor2_view:slice0(int): index "+to_string(i)+" out of range of [0,"+to_string(n0-1)+"]"););
      return Itensor1_view(arr+i*s0,n1,s1,dev);
    }

    Itensor1_view slice1(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n1) throw std::out_of_range("cnine::Itensor2_view:slice1(int): index "+to_string(i)+" out of range of [0,"+to_string(n1-1)+"]"););
      return Itensor1_view(arr+i*s1,n0,s0,dev);
    }

    Itensor1_view fuse01() const{
      return Itensor1_view(arr,n0*n1,s1,dev);
    }

    Itensor2_view block(const int i0, const int i1, int m0=-1, int m1=-1){
      if(m0==-1) m0=n0-i0;
      if(m1==-1) m1=n1-i1;
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=n0 || i1>=n1) 
	  throw std::out_of_range("cnine::Itensor2_view::block: index "+Gindex({i0,i1}).str()+" out of range of size "+Gindex({n0,n1}).str()));
      CNINE_CHECK_RANGE(if(i0+m0<0 || i1+m1<0 || i0+m0>n0 || i1+m1>n1) 
	  throw std::out_of_range("cnine::Itensor2_view::block: end index "+Gindex({i0+m0,i1+m1}).str()+" out of range of size "+Gindex({n0,n1}).str()));
      return Itensor2_view(arr+i0*s0+i1*s1,m0,m1,s0,s1,dev);
    }

    Itensor2_view rows(const int offs, const int n){
      return Itensor2_view(arr+offs*s0,n,n1,s0,s1,dev);
    }

 
  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<int> gtensor() const{
      Gtensor<int> R({n0,n1},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  R(i0,i1)=(*this)(i0,i1);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Itensor2_view& x){
      stream<<x.str(); return stream;
    }

  };


  inline Itensor2_view split0(const Itensor1_view& x, const int i, const int j){
    assert(i*j==x.n0);
    return Itensor2_view(x.arr,i,j,x.s0*j,x.s0,x.dev);
  }



}


#endif 
