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


#ifndef _CnineItensor1_view
#define _CnineItensor1_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"


namespace cnine{


  class Itensor1_view{
  public:

    int* arr;
    int n0;
    int s0;
    int dev=0;

  public:

    Itensor1_view(){}

    Itensor1_view(int* _arr): 
      arr(_arr){}

    Itensor1_view(int* _arr, const int _n0, const int _s0): 
      arr(_arr), n0(_n0), s0(_s0){}

    Itensor1_view(int* _arr, const int _n0, const int _s0, const int _dev): 
      arr(_arr), n0(_n0), s0(_s0), dev(_dev){}

    Itensor1_view(int* _arr,  const Gdims& _dims, const Gstrides& _strides):
      arr(_arr){
      assert(_dims.size()==1);
      n0=_dims[0];
      s0=_strides[0];
    }

    Itensor1_view(int* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a):
      arr(_arr){
      assert(_strides.is_regular(_dims));
      assert(a.is_contiguous());
      assert(a.covers(_dims.size()));
      n0=_dims.unite(a);
      s0=_strides[a.back()];
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    bool is_regular() const{
      if(s0!=1) return false;
      return true;
    }

    int operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::Itensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      CPUCODE(return arr[s0*i0]);
      //GPUCODE(return Itensor_get_cu(arr+s0*i0));
      return 0;
    }

    void set(const int i0, int x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::Itensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      CPUCODE(arr[s0*i0]=x);
      //GPUCODE(Itensor_set_cu(arr+s0*i0,x));
    }

    void inc(const int i0, int x) const{
      CNINE_CHECK_RANGE(if(i0<0 || i0>=n0) 
	  throw std::out_of_range("cnine::Itensor1_view: index "+Gindex({i0}).str()+" out of range of view size "+Gdims({n0}).str()));
      arr[s0*i0]+=x;
      //GPUCODE(Itensor_inc_cu(arr+s0*i0,x));
    }

    Itensor1_view block(const int i0, const int m0) const{
      return Itensor1_view(arr+i0*s0,m0,s0,dev);
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void set(const Itensor1_view& x) const{
      CNINE_DEVICE_SAME(x);
      assert(x.n0==n0);
      if(is_regular() && x.is_regular()){
	CPUCODE(std::copy(x.arr,x.arr+n0,arr));
	GPUCODE(CUDA_SAFE(cudaMemcpy(arr,x.arr,n0*sizeof(int),cudaMemcpyDeviceToDevice)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) set(i0,x(i0)));
	//GPUCODE(CUDA_STREAM(Itensor_copy_cu(*this,x,stream)));
      }
    }


    void add(const Itensor1_view& x) const{
      CNINE_DEVICE_SAME(x);
      assert(x.n0==n0);
      if(is_regular() && x.is_regular()){
	CPUCODE(stdadd<int>(x.arr,x.arr+n0,arr));
	//GPUCODE(const int alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&alpha,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) set(i0,x(i0)));
	//GPUCODE(const int alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&alpha,x.arr,x.s0,arr,s0)));
      }
    }

    void add(const Itensor1_view& x, const int c) const{
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n0);
      if(is_regular() && x.is_regular()){
	CPUCODE(for(int i0=0; i0<n0; i0++) arr[i0]+=c*x.arr[i0];);
	//GPUCODE(CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&c,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<n0; i0++) inc(i0,c*x(i0)));
	//GPUCODE(CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0,&c,x.arr,x.s0,arr,s0)));
      }
    }

    void operator+=(const Itensor1_view& x) const{
      add(x);
    }


  public: // ---- Reductions --------------------------------------------------------------------------------


    int reduce() const{
      int t=0;
      CPUCODE(for(int i=0; i<n0; i++) t+=arr[i*s0]);
      //GPUCODE(CUBLAS_SAFE(cublasSasum(cnine_cublas,n0,arr,s0,&t)));
      return t;
    }


  public: // ---- Broadcasting ------------------------------------------------------------------------------


    void broadcast(const int v){
      assert(is_regular());
      CNINE_CPUONLY();
      CPUCODE(std::fill_n(arr,n0,v));
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<int> gtensor() const{
      Gtensor<int> R({n0},fill::raw);
      for(int i0=0; i0<n0; i0++)
	R(i0)=(*this)(i0);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Itensor1_view& x){
      stream<<x.str(); return stream;
    }


  };



}


#endif 
