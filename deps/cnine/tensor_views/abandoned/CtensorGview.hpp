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


#ifndef _CnineCtensorGview
#define _CnineCtensorGview

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"
//#include "Rmask1.hpp"

//#include "Ctensor1_view.hpp"

//#ifdef _WITH_CUBLAS
//#include <cublas_v2.h>
//extern cublasHandle_t cnine_cublas;
//#endif 


namespace cnine{


  class CtensorGview{
  public:

    float* arr;
    float* arrc;
    Gdims dims;
    Gstrides strides;
    int dev=0;

  public:

    CtensorGview(){}

    CtensorGview(float* _arr, float* _arrc): 
      arr(_arr), arrc(_arrc){}

    CtensorGview(float* _arr, float* _arrc, const int _n0, const int _n1, const int _s0, const int _s1, const int _dev=0): 
      arr(_arr), arrc(_arrc), n0(_n0), n1(_n1), s0(_s0), s1(_s1), dev(_dev){}

    CtensorGview(float* _arr, const int _n0, const int _n1, const int _s0, const int _s1, const int _coffs=1, const int _dev=0): 
      arr(_arr), arrc(_arr+_coffs), n0(_n0), n1(_n1), s0(_s0), s1(_s1), dev(_dev){}

    CtensorGview(float* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_dims.size()==2);
      n0=_dims[0];
      n1=_dims[1];
      s0=_strides[0];
      s1=_strides[1];
    }

    CtensorGview(float* _arr, const Gdims& _dims, const Gstrides& _strides, 
      const GindexSet& a, const GindexSet& b, const int _coffs=1, const int _dev=0):
      arr(_arr), arrc(_arr+_coffs), dev(_dev){
      assert(_strides.is_regular());
      assert(a.is_contiguous());
      assert(b.is_contiguous());
      assert(a.is_disjoint(b));
      assert(a.covers(_dims.size(),b));
      n0=_dims.unite(a);
      n1=_dims.unite(b);
      s0=_strides[a.back()];
      s1=_strides[b.back()];
    }

    

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


    void add_matmul(const CtensorGview& x, const CtensorGview& y){
      add_matmul_AA(x,y);
    }

    void add_matmul_AA(const CtensorGview& x, const CtensorGview& y){
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
	assert(is_regular());
	#ifdef _WITH_CUBLAS
	cuComplex alpha;
	alpha.x=1.0f;
	alpha.y=0.0f;
	#ifndef _OBJFILE
	CUBLAS_SAFE(cublasCgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,n1,n0,x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(y.arr),n1, 
	    reinterpret_cast<cuComplex*>(x.arr),x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(arr),n1)); 
	#endif
	#endif
      }
    }


    void add_matmul_AH(const CtensorGview& x, const CtensorGview& y){
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
	    reinterpret_cast<cuComplex*>(y.arr),n1, 
	    reinterpret_cast<cuComplex*>(x.arr),x.n1,&alpha,
	    reinterpret_cast<cuComplex*>(arr),n1)); 
	#endif
      }
    }



/*
    void accumulate(const CtensorGview& x, const Rmask1& mask){
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




    CtensorGview transp(){
      CtensorGview R(arr,arrc);
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

    friend ostream& operator<<(ostream& stream, const CtensorGview& x){
      stream<<x.str(); return stream;
    }


  };

  inline CtensorGview split0(const Ctensor1_view& x, const int i, const int j){
    assert(i*j==x.n0);
    return CtensorGview(x.arr,x.arrc,i,j,x.s0*j,x.s0,x.dev);
  }


  inline CtensorGview unsqueeze0(const Ctensor1_view& x){
    return CtensorGview(x.arr,x.arrc,1,x.n0,x.s0*x.n0,x.s0,x.dev);
  }

  inline CtensorGview unsqueeze1(const Ctensor1_view& x){
    return CtensorGview(x.arr,x.arrc,x.n0,1,x.s0,x.s0,x.dev);
  }





}


#endif 
