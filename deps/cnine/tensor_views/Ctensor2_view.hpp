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


#ifndef _CnineCtensor2_view
#define _CnineCtensor2_view

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"
//#include "Rmask1.hpp"

#include "Ctensor1_view.hpp"
#include "Rtensor2_view.hpp"
//#include "TensorView.hpp"

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


//#ifdef _WITH_CUDA
//class Ctensor2_view; 
//void Ctensor2view_accumulator_cu(Ctensor2_view& r, const Ctensor2_view& x, const Rmask1& mask, const cudaStream_t& stream);
//#endif 


namespace cnine{

  #ifdef _WITH_CUDA
  extern void Rtensor_add_cu(const Rtensor2_view& r, const Rtensor2_view& x, const cudaStream_t& stream);
  //extern void Rtensor_add_cu(const Rtensor2_view& r, const Rtensor2_view& x, const float c, const cudaStream_t& stream);
  //extern void Rtensor_sum0_into_cu(const Rtensor1_view& r, const Rtensor2_view& x, const cudaStream_t& stream);
  //extern void Rtensor_sum0_into_cu(const Rtensor1_view& r, const Rtensor2_view& x, const float c, const cudaStream_t& stream);
  #endif 

  class Ctensor2_view{
  public:

    float* arr;
    float* arrc;
    int n0,n1;
    int s0,s1;
    int dev=0;

  public:

    Ctensor2_view(){}

    Ctensor2_view(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    Ctensor2_view(float* _arr, float* _arrc, const int _n0, const int _n1, const int _s0, const int _s1, const int _dev=0): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), s0(_s0), s1(_s1), dev(_dev){}

    Ctensor2_view(float* _arr, const int _n0, const int _n1, const int _s0, const int _s1, const int _coffs=1, const int _dev=0): 
      arr(_arr), arrc(_arr+_coffs), n0(_n0), n1(_n1), s0(_s0), s1(_s1), dev(_dev){}

    Ctensor2_view(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_dims.size()==2);
      n0=_dims[0];
      n1=_dims[1];
      s0=_strides[0];
      s1=_strides[1];
    }

    Ctensor2_view(float* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a, const GindexSet& b, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
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

//     #ifdef _CnineTensorViewComplete
//     Ctensor2_view(const TensorView<complex<float> >& x):
//       arr(x.arr.ptr_as<float>()),
//       arrc(x.arr.ptr_as<float>()+1),
//       dev(x.dev){
//       CNINE_ASSRT(x.ndims()==2);
//       n0=x.dim(0);
//       n1=x.dim(1);
//       s0=2*x.strides[0];
//       s1=2*x.strides[1];
//     }
//     #endif 


  public: // ---- Access ------------------------------------------------------------------------------------


    complex<float> operator()(const int i0, const int i1) const{
      int t=s0*i0+s1*i1;
      return complex<float>(arr[t],arrc[t]);
    }

    void set(const int i0, const int i1, complex<float> x) const{
      int t=s0*i0+s1*i1;
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, complex<float> x) const{
      int t=s0*i0+s1*i1;
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }

    bool is_regular() const{
      if(arrc-arr!=1) return false;
      if(s1!=2) return false;
      if(s0!=s1*n1) return false;
      return true;
    }


  public: // ---- Cumulative operations ---------------------------------------------------------------------


    void add(const Ctensor2_view& x) const{
      CNINE_UNIMPL();
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.n0==n0);
      CNINE_ASSRT(x.n1==n1);
      if(x.n0*x.n1==0) return;
      if(is_regular() && x.is_regular()){
	CPUCODE(stdadd<float>(x.arr,x.arr+n0*s0,arr));
	GPUCODE(const float alpha=1; CUBLAS_SAFE(cublasSaxpy(cnine_cublas,n0*n1,&alpha,x.arr,1,arr,1)));
      }else{
	CPUCODE(for(int i0=0; i0<x.n0; i0++) for(int i1=0; i1<x.n1; i1++) {inc(i0,i1,x(i0,i1));});
	CNINE_CPUONLY();
	//GPUCODE(CUDA_STREAM(Rtensor_add_cu(*this,x,stream)));
      }
    }

    void add_matmul(const Ctensor2_view& x, const Ctensor2_view& y){
      add_matmul_AA(x,y);
    }

    void add_matmul_AA(const Ctensor2_view& x, const Ctensor2_view& y){
      CNINE_DEVICE_SAME(x);
      CNINE_DEVICE_SAME(y);
      const int I=x.n1;

      if(dev==0){
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++){
	    complex<float> t=0;
	    for(int i=0; i<I; i++)
	      t+=x(a,i)*y(i,b);
	    inc(a,b,t);
	  }
      }

      if(dev==1){
	assert(is_regular());
	#ifdef _WITH_CUBLAS
	cuComplex alpha;
	alpha.x=1.0f;
	alpha.y=0.0f;
	#ifndef _OBJFILE
	CUBLAS_SAFE(cublasCgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,n1,n0,x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(y.arr),y.n1, 
	    reinterpret_cast<cuComplex*>(x.arr),x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(arr),n1)); 
	#endif
	#endif
      }
    }

    void add_mprod(const Ctensor2_view& x, const Ctensor2_view& y){
      add_matmul_AA(x,y);
    }

    void add_matmul_AH(const Ctensor2_view& x, const Ctensor2_view& y){
      CNINE_CHECK_DEV3((*this),x,y);
      assert(x.n0==n0);
      assert(y.n0==n1);
      assert(y.n1==x.n1);
      const int I=x.n1;


      if(dev==0){
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++){
	    complex<float> t=0;
	    for(int i=0; i<I; i++)
	      t+=x(a,i)*std::conj(y(b,i));
	    inc(a,b,t);
	  }
      }

      if(dev==1){
	assert(is_regular());
	#ifdef _WITH_CUBLAS
	cuComplex alpha;
	alpha.x=1.0f;
	alpha.y=0.0f;
	CUBLAS_SAFE(cublasCgemm(cnine_cublas,CUBLAS_OP_C,CUBLAS_OP_N,n1,n0,x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(y.arr),y.n1, 
	    reinterpret_cast<cuComplex*>(x.arr),x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(arr),n1)); 
	#endif
      }
    }



    void add_matmul_HA(const Ctensor2_view& x, const Ctensor2_view& y){
      CNINE_CHECK_DEV3((*this),x,y);
      assert(x.n1==n0);
      assert(y.n1==n1);
      assert(y.n0==x.n0);
      const int I=x.n0;


      if(dev==0){
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++){
	    complex<float> t=0;
	    for(int i=0; i<I; i++)
	      t+=std::conj(x(i,a))*y(i,b);
	    inc(a,b,t);
	  }
      }

      if(dev==1){
	assert(is_regular());
	#ifdef _WITH_CUBLAS
	cuComplex alpha;
	alpha.x=1.0f;
	alpha.y=0.0f;
	CUBLAS_SAFE(cublasCgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_C,n1,n0,x.n0,&alpha,
	    reinterpret_cast<cuComplex*>(y.arr),y.n1, 
	    reinterpret_cast<cuComplex*>(x.arr),x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(arr),n1)); 
	#endif
      }
    }



    void add_matmul_NA(const Rtensor2_view& x, const Ctensor2_view& y){
      CNINE_CHECK_DEV3((*this),x,y);
      assert(x.n0==n0);
      assert(y.n1==n1);
      assert(y.n0==x.n1);
      const int I=x.n1;


      if(dev==0){
	for(int a=0; a<n0; a++)
	  for(int b=0; b<n1; b++){
	    complex<float> t=0;
	    for(int i=0; i<I; i++)
	      t+=x(a,i)*y(i,b);
	    inc(a,b,t);
	  }
      }

      if(dev==1){
	CNINE_UNIMPL();
	assert(is_regular());
	
	#ifdef _WITH_CUBLAS
	cuComplex alpha;
	alpha.x=1.0f;
	alpha.y=0.0f;
	#ifndef _OBJFILE
	CUBLAS_SAFE(cublasCgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,n1,n0,x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(y.arr),y.n1, 
	    reinterpret_cast<cuComplex*>(x.arr),x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(arr),n1)); 
	#endif
	#endif
      }
    }




/*
    void accumulate(const Ctensor2_view& x, const Rmask1& mask){
      if(dev==0){
	assert(x.dev==0);
	for(auto it: mask.lists){
	  auto t=slice0(it.first);
	  auto& lst=it.second;
	  for(int i=0; i<lst.size(); i++)
	    t.add(x.slice0(lst[i].first),lst[i].second);
	}
      }
      if(dev==1){
#ifdef _WITH_CUDA
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	Ctensor2view_accumulator_cu(*this,x,mask,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
      }
      
    }
  */  

  public: // ---- Other views -------------------------------------------------------------------------------


    Ctensor1_view slice0(const int i) const{
      return Ctensor1_view(arr+i*s0,arrc+i*s0,n1,s1,dev);
    }

    Ctensor1_view slice1(const int i) const{
      return Ctensor1_view(arr+i*s1,arrc+i*s1,n0,s0,dev);
    }
    
    Ctensor1_view fuse01() const{
      return Ctensor1_view(arr,arrc,n0*s0,s1,dev);
    }




    Ctensor2_view transp(){
      Ctensor2_view R(arr,arrc);
      R.n0=n1;
      R.n1=n0;
      R.s0=s1;
      R.s1=s0;
      return R;
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    Gtensor<complex<float> > gtensor() const{
      Gtensor<complex<float> > R({n0,n1},fill::raw);
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  R(i0,i1)=(*this)(i0,i1);
      return R;
    }
    
    
  public: // ---- I/O ----------------------------------------------------------------------------------------

  
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Ctensor2_view& x){
      stream<<x.str(); return stream;
    }


  };

  inline Ctensor2_view split0(const Ctensor1_view& x, const int i, const int j){
    assert(i*j==x.n0);
    return Ctensor2_view(x.arr,x.arrc,i,j,x.s0*j,x.s0,x.dev);
  }


  inline Ctensor2_view unsqueeze0(const Ctensor1_view& x){
    return Ctensor2_view(x.arr,x.arrc,1,x.n0,x.s0*x.n0,x.s0,x.dev);
  }

  inline Ctensor2_view unsqueeze1(const Ctensor1_view& x){
    return Ctensor2_view(x.arr,x.arrc,x.n0,1,x.s0,x.s0,x.dev);
  }





}


#endif 
    /*
    Ctensor2_view(const Gdims& dims, const Gstrides& strides, const int _coffs=1){
      assert(dims.size()==2);
      n0=dims(0);
      n1=dims(1);
      assert(strides.size()==2);
      s0=strides[0];
      s1=strides[1];
      arrc=arr+_coffs;
    }
    */
