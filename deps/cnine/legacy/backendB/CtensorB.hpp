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


#ifndef _CnineCtensorB
#define _CnineCtensorB

#include "Cnine_base.hpp"
#include "CnineObject.hpp"
#include "RscalarA.hpp"
#include "CscalarA.hpp"
#include "RtensorA.hpp"
#include "Gdims.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"
#include "CtensorB_accessor.hpp"

#include "Ctensor1_view.hpp"
#include "Ctensor2_view.hpp"
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"
#include "Ctensor5_view.hpp"
#include "Ctensor6_view.hpp"
#include "CtensorView.hpp"
#include "CtensorD_view.hpp"

#include "CtensorView.hpp"
#include "Aggregator.hpp"

#include "Ctensor_mprodFn.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuComplex.h"
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{


  class CtensorB: public CnineObject{
  public:

    Gdims dims;
    Gstrides strides;

    int asize=0;
    int memsize=0;
    int coffs=0; 

    int dev=0;
    bool is_view=false;

    float* arr=nullptr;
    float* arrg=nullptr;


  public:

    CtensorB(){}

    ~CtensorB(){
      if(!is_view && arr) {delete[] arr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
    }

    string classname() const{
      return "CtensorB";
    }

    string describe() const{
      return "CtensorB"+dims.str();
    }


  public: // ---- Constructors -----------------------------------------------------------------------------

    
    CtensorB(const Gdims& _dims, const Gstrides& _strides, 
      const int _asize, const int _memsize, const int _coffs, const int _dev):
     dims(_dims), strides(_strides), asize(_asize), memsize(_memsize), coffs(_coffs), dev(_dev){}


    // Root constructor: elements are uninitialized
    CtensorB(const Gdims& _dims, const int _dev=0): 
      dims(_dims), dev(_dev), strides(_dims,2){

      CNINE_DIMS_VALID(dims);
      CNINE_DEVICE_VALID(dev);

      asize=strides[0]*dims[0]/2; 
      memsize=strides[0]*dims[0]; 
      coffs=1;

      if(dev==0){
	arr=new float[memsize];
      }

      if(dev==1){
	CNINE_REQUIRES_CUDA()
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
      }

    }


  public: // ---- Filled constructors -----------------------------------------------------------------------

    
    CtensorB(const Gdims& _dims, const fill_noalloc& dummy, const int _dev=0): 
      dims(_dims), dev(_dev), strides(_dims,2){
      CNINE_DEVICE_VALID(dev)
      asize=strides[0]*dims[0]/2; 
      memsize=strides[0]*dims[0]; 
      coffs=1;
    }

    CtensorB(const Gdims& _dims, const fill_raw& dummy, const int _dev=0): 
      CtensorB(_dims,_dev){}
    
    CtensorB(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      CtensorB(_dims,_dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    CtensorB(const Gdims& _dims, const fill_ones& dummy, const int _dev=0): 
      CtensorB(_dims,fill::raw){
      std::fill(arr,arr+memsize,1);
      if(_dev>0) move_to_device(_dev);
    }

    CtensorB(const Gdims& _dims, const fill_identity& dummy, const int _dev=0): 
      CtensorB(_dims,fill_zero()){
      CNINE_NDIMS_IS_2((*this));
      assert(dims[0]==dims[1]);
      int s=strides[0]+strides[1];
      for(int i=0; i<dims[0]; i++)
	arr[s*i]=1;
      if(_dev>0) move_to_device(_dev);
    }

    CtensorB(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      CtensorB(_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<memsize; i++) arr[i]=distr(rndGen);
      if(_dev>0) move_to_device(_dev);
    }
    
    CtensorB(const Gdims& _dims, const fill_gaussian& dummy, const float c, const int _dev):
      CtensorB(_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<memsize; i++) arr[i]=c*distr(rndGen);
      if(_dev>0) move_to_device(_dev);
    }

    CtensorB(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      CtensorB(_dims,fill::zero,0){
      int s=strides[getk()-1];
      for(int i=0; i<asize; i++) arr[i*s]=i;
      if(_dev>0) move_to_device(_dev);
    }
	  

  public: // ---- Named constructors -------------------------------------------------------------------------


    static CtensorB noalloc(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_noalloc(),_dev);
    }

    static CtensorB raw(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_raw(),_dev);
    }

    static CtensorB zero(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_zero(),_dev);
    }

    static CtensorB zeros(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_zero(),_dev);
    }

    static CtensorB ones(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_ones(),_dev);
    }

    static CtensorB sequential(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_sequential(),_dev);
    }

    static CtensorB gaussian(const Gdims& _dims, const int _dev=0){
      return CtensorB(_dims,fill_gaussian(),_dev);
    }


    template<typename FILL>
    static CtensorB like(const CtensorB& x){
      return CtensorB(x.dims,FILL(),x.dev);
    }

    static CtensorB raw_like(const CtensorB& x){
      return CtensorB(x.dims,fill_raw(),x.dev);
    }

    static CtensorB zeros_like(const CtensorB& x){
      return CtensorB(x.dims,fill_zero(),x.dev);
    }

    static CtensorB view_of_array(const Gdims& _dims, float* _arr, const int _dev=0){
      CtensorB R=CtensorB::noalloc(_dims,_dev);
      if(_dev==0) R.arr=_arr;
      if(_dev==1) R.arrg=_arr;
      R.is_view=true;
      return R;
    }

    static const CtensorB view_of_array(const Gdims& _dims, const float* _arr, const int _dev=0){
      CtensorB R=CtensorB::noalloc(_dims,_dev);
      if(_dev==0) R.arr=const_cast<float*>(_arr);
      if(_dev==1) R.arrg=const_cast<float*>(_arr);
      R.is_view=true;
      return R;
    }


  public: // ---- Spawning -----------------------------------------------------------------------------------


    static CtensorB* new_zeros_like(const CtensorB& x){
      return new CtensorB(x.dims,fill_zero(),x.dev);
    }


  public: // ---- Lambda constructors -------------------------------------------------------------------------



  public: // ---- Copying -----------------------------------------------------------------------------------


    CtensorB(const CtensorB& x): 
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,x.dev){
      CNINE_COPY_WARNING();
      if(dev==0){
	arr=new float[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CNINE_REQUIRES_CUDA();
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
    }
        
    CtensorB(const CtensorB& x, const nowarn_flag& dummy): 
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,x.dev){
      if(dev==0){
	arr=new float[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CNINE_REQUIRES_CUDA();
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
    }
        
    CtensorB(const CtensorB& x, const int _dev): 
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,_dev){
      if(dev==0){
	if(x.dev==0){
	  arr=new float[memsize];
	  std::copy(x.arr,x.arr+memsize,arr);
	}
	if(x.dev==1){
	  CNINE_REQUIRES_CUDA();
	  arr=new float[memsize];
	  CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost)); 
	}
      }
      if(dev==1){
	CNINE_REQUIRES_CUDA();
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	if(x.dev==0){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(float),cudaMemcpyHostToDevice));
	}
	if(x.dev==1){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	}
      }
    }

    CtensorB(const CtensorB& x, const view_flag& dummy):
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,x.dev){
      arr=x.arr;
      arrg=x.arrg;
      is_view=true;
    }
        
    CtensorB(CtensorB&& x): 
      CtensorB(x.dims,x.strides,x.asize,x.memsize,x.coffs,x.dev){
      CNINE_MOVE_WARNING();
      arr=x.arr; x.arr=nullptr; 
      arrg=x.arrg; x.arrg=nullptr;
      is_view=x.is_view;
    }

    CtensorB* clone() const{
      return new CtensorB(*this);
    }

    CtensorB& operator=(const CtensorB& x){
      CNINE_ASSIGN_WARNING();
      if(this==&x) return *this;
      if(!is_view && arr) {delete arr; arr=nullptr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg)); arrg=nullptr;}
      dims=x.dims; 
      strides=x.strides; 
      asize=x.asize; 
      memsize=x.memsize; 
      coffs=x.coffs;
      dev=x.dev;

      if(dev==0){
	arr=new float[memsize]; 
	std::copy(x.arr,x.arr+memsize,arr);
      }

      if(dev==1){
	CNINE_REQUIRES_CUDA();
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
      
      is_view=false;
      return *this;
    }


    CtensorB& operator=(CtensorB&& x){
      CNINE_MOVEASSIGN_WARNING();
      if(this==&x) return *this;
      if(!is_view && arr) {delete arr; arr=nullptr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg)); arrg=nullptr;}
      dims=x.dims; 
      strides=x.strides; 
      asize=x.asize; 
      memsize=x.memsize; 
      coffs=x.coffs; 
      dev=x.dev; 
      arr=x.arr; x.arr=nullptr; 
      arrg=x.arrg; x.arrg=nullptr; 
      is_view=x.is_view;
      return *this;
    }


  public: // ---- Views -------------------------------------------------------------------------------


    CtensorB view(){
      CtensorB R=CtensorB::noalloc(dims,dev);
      R.arr=arr;
      R.arrg=arrg;
      R.is_view=true;
      return R;
    }


    CtensorB* viewp(){
      CtensorB* R=new CtensorB(dims,fill_noalloc(),dev);
      R->arr=arr;
      R->arrg=arrg;
      R->is_view=true;
      return R;
    }


    CtensorB view_as_shape(const Gdims& _dims){
      CNINE_DIMS_EQ_TOTAL(dims,_dims);
      CtensorB R=CtensorB::noalloc(_dims,dev);
      R.arr=arr;
      R.arrg=arrg;
      R.is_view=true;
      return R;
    }

    CtensorB view_fusing_first(const int k){
      assert(k<dims.size());
      int c=dims.size()-k;
      int t=1;
      for(int i=0; i<k; i++)
	t*=dims[i];
      Gdims _dims(c+1,fill_raw());
      _dims[0]=t;
      for(int i=0; i<c; i++)
	_dims[i+1]=dims[i+k];
      CtensorB R=CtensorB::noalloc(_dims,dev);
      R.arr=arr;
      R.arrg=arrg;
      R.is_view=true;
      return R;      
    }

    /*
    const CtensorB view_fusing_first(const int k) const{
      assert(k<dims.size());
      int c=dims.size()-k;
      int t=1;
      for(int i=0; i<k; i++)
	t*=dims[i];
      Gdims _dims(c+1,fill_raw());
      _dims[0]=t;
      for(int i=0; i<c; i++)
	_dims[i+1]=dims[i+k];
      CtensorB R=CtensorB::noalloc(_dims,dev);
      R.arr=arr;
      R.arrg=arrg;
      R.is_view=true;
      return R;      
    }
    */


  public: // ---- Conversions -------------------------------------------------------------------------------


    CtensorB(const RtensorA& re, const RtensorA& im):
      CtensorB(re.dims,fill_raw(),re.dev){
      CNINE_DIMS_SAME(im);
      CNINE_DEVICE_SAME(im);
      if(dev==0){
	for(int i=0; i<asize; i++)
	  arr[2*i]=re.arr[i];
	for(int i=0; i<asize; i++)
	  arr[2*i+1]=im.arr[i];
      }
      if(dev==1){
	CNINE_CPUONLY();
      }
    }


    RtensorA real() const{
      RtensorA R=RtensorA::raw(dims,dev);
      if(dev==0){
	for(int i=0; i<asize; i++)
	  R.arr[i]=arr[2*i];
      }
      if(dev==1){
	CNINE_CPUONLY();
      }
      return R;
    }


    RtensorA imag() const{
      RtensorA R=RtensorA::raw(dims,dev);
      if(dev==0){
	for(int i=0; i<asize; i++)
	  R.arr[i]=arr[2*i+1];
      }
      if(dev==1){
	CNINE_CPUONLY();
      }
      return R;
    }


  public: // ---- Transport -----------------------------------------------------------------------------------


    CtensorB& move_to_device(const int _dev){
      
      if(dev==_dev) return *this;
      if(is_view) throw std::runtime_error("Cnine error in "+string(__PRETTY_FUNCTION__)+": a tensor view cannot be moved to a different device.");
      
      if(_dev==0){
	if(dev==0) return *this;
 	delete[] arr;
	arr=new float[memsize];
	CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arrg));
	const_cast<CtensorB*>(this)->arrg=nullptr;
	dev=0;
	return *this;
      }

      if(_dev>0){
	if(dev==_dev) return *this;
	if(arrg) CUDA_SAFE(cudaFree(arrg));
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(float),cudaMemcpyHostToDevice));  
	delete[] arr;
	const_cast<CtensorB*>(this)->arr=nullptr;
	dev=_dev;
	return *this;
      }
      
      return *this;
    }
    
    CtensorB& move_to(const device& _dev){
      return move_to_device(_dev.id());
    }
    
    CtensorB to(const device& _dev) const{
      return CtensorB(*this,_dev.id());
    }

    CtensorB to_device(const int _dev) const{
      return CtensorB(*this,_dev);
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    CtensorB(const Gtensor<complex<float> >& x, const int _dev=0): 
      CtensorB(x.dims,fill::raw){
      assert(x.dev==0);
      int s=strides[getk()-1];
      for(int i=0; i<asize; i++){
	arr[s*i]=std::real(x.arr[i]);
	arr[s*i+coffs]=std::imag(x.arr[i]);
      }
      move_to_device(_dev);
    }
    
    Gtensor<complex<float> > gtensor() const{
      if(dev>0) return CtensorB(*this,0).gtensor();
      Gtensor<complex<float> > R(dims,fill::raw);
      assert(dev==0);
      int s=strides[getk()-1];
      for(int i=0; i<asize; i++){
	R.arr[i]=complex<float>(arr[s*i],arr[s*i+coffs]);
      }
      return R;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    bool has_same_strides_as(const at::Tensor& T) const{
      int k=strides.size();
      if(T.dim()!=k) return false;
      for(int i=0; i<k; i++)
	if(T.stride(i)!=strides[i]/2) return false;
      //if(T.stride(k)!=2) return false;
      return true;
    }

    static bool is_regular(const at::Tensor& T){
      int k=T.dim();
      Gdims Tdims(k,fill_raw());
      for(int i=0; i<k ; i++)
	Tdims[i]=T.size(i);
      if(Tdims.asize()==0) return true;
      Gstrides Tstrides(Tdims);
      bool t=true;
      for(int i=0; i<k; i++)
	if(Tstrides[i]!=T.stride(i) && Tdims[i]>1) {t=false; break;}
      if(t==false){
	CoutLock lk;
	cout<<"Warning: ATen tensor of dims "<<Tdims<<" has strides [ ";
	for(int i=0; i<k; i++) cout<<T.stride(i)<<" ";
	cout<<"] instead of "<<Tstrides.str()<<endl;
	return false; 
      }
      return true;
    }


    CtensorB(const int dummy, const at::Tensor& T){ // deprecated 
      CNINE_CONVERT_FROM_ATEN_WARNING();
      //assert(T.dtype()==at::kFloat);
      //assert(T.dtype()==at::kComplexFloat);
      T.contiguous();

      int k=T.dim();
      //if(k<=0 || T.size(k)!=2) 
      //throw std::out_of_range("CtensorB: last dimension of ATen tensor must be 2, corresponding to the real and imaginary parts.");
      dims=Gdims::raw(k);
      for(int i=0; i<k ; i++)
	dims[i]=T.size(i);
      strides=Gstrides(dims,2);
      asize=strides[0]*dims[0]/2; 
      memsize=strides[0]*dims[0]; 
      coffs=1;
      dev=T.type().is_cuda();

      if(false && !has_same_strides_as(T)){
	if(dev!=0) 
	  throw std::out_of_range("CtensorB: ATen tensor is irregular and is on the GPU."); 
	auto src=T.data<float>();
	arr=new float[memsize];
	Gstrides sstrides(k+1,fill_raw());
	Gstrides xstrides(strides); 
	xstrides.push_back(2);
	for(int i=0; i<k+1; i++) sstrides[i]=T.stride(i);
	for(int i=0; i<memsize; i++)
	  arr[i]=src[sstrides.offs(i,xstrides)];
      }

      if(dev==0){
	arr=new float[memsize];
	std::copy(T.data<c10::complex<float> >(),T.data<c10::complex<float> >()+asize,reinterpret_cast<c10::complex<float>*>(arr));
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,T.data<complex<float> >(),memsize*sizeof(float),cudaMemcpyDeviceToDevice));
      }

    }


    CtensorB(const at::Tensor& T){
      CNINE_CONVERT_FROM_ATEN_WARNING();
      //assert(T.dtype()==at::kComplexFloat);
      //assert(typeid(T.type().scalarType())==typeid(float));
      T.contiguous();

      int k=T.dim();
      //if(k<=0 || T.size(k)!=2) 
      //throw std::out_of_range("CtensorB: last dimension of ATen tensor must be 2, corresponding to the real and imaginary parts.");
      dims=Gdims::raw(k);
      for(int i=0; i<k ; i++)
	dims[i]=T.size(i);
      strides=Gstrides(dims,2);
      asize=strides[0]*dims[0]/2; 
      memsize=strides[0]*dims[0]; 
      coffs=1;
      dev=T.type().is_cuda();

      if(!has_same_strides_as(T)){
	cout<<"irregular strides"<<endl;
	if(dev!=0) 
	  throw std::out_of_range("CtensorB: ATen tensor is irregular and is on the GPU."); 
	auto src=T.data<c10::complex<float> >();
	arr=new float[memsize];
	//Gstrides xstrides(strides); 
	//xstrides.push_back(2);
	Gstrides sstrides(k,fill_raw());
	for(int i=0; i<k; i++) sstrides[i]=T.stride(i);
	for(int i=0; i<memsize; i+=2)
	  *reinterpret_cast<c10::complex<float>*>(arr+i*2)=src[sstrides.offs(i,strides)];
      }

      if(dev==0){
	arr=new float[memsize];
	std::copy(T.data<c10::complex<float> >(),T.data<c10::complex<float> >()+asize,reinterpret_cast<c10::complex<float>*>(arr));
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,T.data<c10::complex<float> >(),memsize*sizeof(float),cudaMemcpyDeviceToDevice));
      }

    }

    static CtensorB view(at::Tensor& T){
      T.contiguous();
      if(!is_regular(T)){cout<<"irregular!"<<endl; /*return CtensorB(T);*/}
      
      CtensorB R;
      int k=T.dim(); //-1;
      //if(k<=0 || T.size(k)!=2) throw std::out_of_range("CtensorB: last dimension of tensor must be 2, corresponding to the real and imaginary parts.");
      R.dims.resize(k);
      for(int i=0; i<k ; i++)
	R.dims[i]=T.size(i);
      R.strides=Gstrides(R.dims,2);
      R.asize=R.strides[0]*R.dims[0]/2; 
      for(int i=0; i<k; i++)
	R.strides[i]=2*T.stride(i); // changed 
      R.memsize=R.strides[0]*R.dims[0]; 
      R.coffs=1;
      R.dev=T.type().is_cuda();
      R.is_view=true;

      if(R.dev==0){
	R.arr=reinterpret_cast<float*>(T.data<c10::complex<float> >());
      }
      
      if(R.dev==1){
	R.arrg=reinterpret_cast<float*>(T.data<c10::complex<float> >());
      }
      return R;
    }

    static CtensorB* viewp(at::Tensor& T){
      T.contiguous();
      if(!is_regular(T)){cout<<"irregular!"<<endl; return new CtensorB(T);}
      
      CtensorB* R=new CtensorB();
      int k=T.dim(); //-1;
      //if(k<=0 || T.size(k)!=2) throw std::out_of_range("CtensorB: last dimension of tensor must be 2, corresponding to the real and imaginary parts.");
      R->dims.resize(k);
      for(int i=0; i<k ; i++)
	R->dims[i]=T.size(i);
      R->strides=Gstrides(R->dims,2);
      R->asize=R->strides[0]*R->dims[0]/2; 
      R->memsize=R->strides[0]*R->dims[0]; 
      R->coffs=1;
      R->dev=T.type().is_cuda();
      R->is_view=true;

      if(R->dev==0){
	R->arr=reinterpret_cast<float*>(T.data<c10::complex<float> >());
	//R->arr=T.data<float>();
      }
      
      if(R->dev==1){
	R->arrg=reinterpret_cast<float*>(T.data<c10::complex<float> >());
	//R->arrg=T.data<float>();
      }

      return R;
    }

    at::Tensor torch() const{
      CNINE_CONVERT_TO_ATEN_WARNING();
      assert(dev==0);
      assert(coffs==1);
      int k=getk();
      vector<int64_t> v(k); 
      for(int i=0; i<k; i++) v[i]=dims[i];
      //v[k]=2;
      at::Tensor R(at::zeros(v,torch::CPU(at::kComplexFloat))); 
      std::copy(arr,arr+memsize,reinterpret_cast<float*>(R.data<c10::complex<float> >()));
      return R;
    }

    /*
    at::Tensor move_to_torch(){ // TODO 
      CNINE_CONVERT_TO_ATEN_WARNING();
      assert(dev==0);
      vector<int64_t> v(k+1); 
      for(int i=0; i<k; i++) v[i+1]=dims[i];
      v[0]=2;
      at::Tensor R(at::zeros(v,torch::CPU(at::kFloat))); 
      std::copy(arr,arr+asize,R.data<float>());
      std::copy(arrc,arrc+asize,R.data<float>()+asize);
      return R;
    }
    */

#endif 



  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nbu() const{ // for backward compatibility 
      return -1;
    }

    int getk() const{
      return dims.size();
    }

    int ndims() const{
      return dims.size();
    }

    Gdims get_dims() const{
      return dims;
    }

    int get_dim(const int i) const{
      return dims[i];
    }

    int dim(const int i) const{
      return dims[i];
    }

    int get_dev() const{
      return dev;
    }

    int get_device() const{
      return dev;
    }

    float* get_arr() const{
      if(dev==0) return arr;
      else return arrg;
    }

    float* true_arr() const{
      if(dev==0) return arr;
      else return arrg;
    }


  public: // ---- Setting ------------------------------------------------------------------------------------


    void set(const CtensorB& x){
      CNINE_DEVICE_SAME(x);
      CNINE_DIMS_SAME(x);
      if(dev==0){
	std::copy(x.arr,x.arr+memsize,arr);
	return; 
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }


    void set_zero(){
      if(dev==0){
	std::fill(arr,arr+memsize,0);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
      }
    }


  public: // ---- Accessors ----------------------------------------------------------------------------------

    
    //CtensorB_accessor1 access_as_1D() const{
    //return CtensorB_accessor1(strides);
    //}

    //CtensorB_accessor2 access_as_2D() const{
    //return CtensorB_accessor2(strides);
    //}

    //CtensorB_accessor3 access_as_3D() const{
    //return CtensorB_accessor3(strides);
    //}


  public: // ---- Access views --------------------------------------------------------------------------------


    Ctensor1_view view1() const{
      if(dev==0) return Ctensor1_view(arr,dims,strides,coffs,dev);
      else return Ctensor1_view(arrg,dims,strides,coffs,dev);
    }

    Ctensor1_view view1D() const{
      return Ctensor1_view(arr,dims,strides,coffs,dev);
    }

    Ctensor1_view view1D(const GindexSet& a) const{
      return Ctensor1_view(arr,dims,strides,a,coffs);
    }


    Ctensor2_view view2() const{
      if(dev==0) return Ctensor2_view(arr,dims,strides,coffs,dev);
      else return Ctensor2_view(arrg,dims,strides,coffs,dev);
    }

    Ctensor2_view view2D() const{
      return Ctensor2_view(arr,dims,strides,coffs,dev);
    }

    Ctensor2_view view2D(const GindexSet& a, const GindexSet& b) const{
      return Ctensor2_view(arr,dims,strides,a,b,coffs);
    }


    Ctensor3_view view3() const{
      if(dev==0) return Ctensor3_view(arr,dims,strides,coffs,dev);
      else return Ctensor3_view(arrg,dims,strides,coffs,dev);
    }

    Ctensor3_view view3D() const{
      return Ctensor3_view(true_arr(),dims,strides,coffs,dev);
    }

    Ctensor3_view view3D(const GindexSet& a, const GindexSet& b, const GindexSet& c) const{
      return Ctensor3_view(arr,dims,strides,a,b,c,coffs);
    }


    const Ctensor4_view view4() const{
      return Ctensor4_view(true_arr(),dims,strides,coffs,dev);
    }

    Ctensor4_view view4(){
      return Ctensor4_view(true_arr(),dims,strides,coffs,dev);
    }


    const Ctensor5_view view5() const{
      return Ctensor5_view(true_arr(),dims,strides,coffs,dev);
    }

    Ctensor5_view view5(){
      return Ctensor5_view(true_arr(),dims,strides,coffs,dev);
    }


    const Ctensor6_view view6() const{
      return Ctensor6_view(true_arr(),dims,strides,coffs,dev);
    }

    Ctensor6_view view6(){
      return Ctensor6_view(true_arr(),dims,strides,coffs,dev);
    }


    CtensorView viewx(){
      return CtensorView(true_arr(),true_arr()+coffs,dims,strides,dev);
    }

    const CtensorView viewx() const{
      return CtensorView(true_arr(),true_arr()+coffs,dims,strides,dev);
    }

    CtensorD_view viewd(){
      return CtensorD_view(true_arr(),true_arr()+coffs,dims,strides,dev);
    }

    const CtensorD_view viewd() const{
      return CtensorD_view(true_arr(),true_arr()+coffs,dims,strides,dev);
    }


    Ctensor2_view pick_dimension(const int ix=0){
      int k=getk();
      assert(k>=2);
      Ctensor2_view r;

      if(ix==0){
	r.n0=dims(0);
	r.n1=strides[0]/strides[k-1];
	r.s0=strides[0];
	r.s1=strides[k-1];
	r.arr=arr;
	r.arrc=arr+coffs;
	return r;
      }

      if(ix==k-1){
	r.n0=dims(k-1);
	r.n1=asize/dims(k-1);
	r.s0=strides[k-1];
	r.s1=strides[k-2];
	r.arr=arr;
	r.arrc=arr+coffs;
	return r;
      }

      assert(false);
      return r;
    }

    
  public: // ---- Element Access ------------------------------------------------------------------------------
    

    complex<float> operator()(const Gindex& ix) const{
      int t=ix(strides);  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> operator()(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3] || i4<0 || i4>=dims[4]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3,i4).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4];  
      return complex<float>(arr[t],arr[t+coffs]);
    }


    complex<float> get(const Gindex& ix) const{
      int t=ix(strides);  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get(const int i0) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3] || i4<0 || i4>=dims[4]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3,i4).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4];  
      return complex<float>(arr[t],arr[t+coffs]);
    }


    complex<float> get_value(const Gindex& ix) const{
      int t=ix(strides);  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get_value(const int i0) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get_value(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get_value(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get_value(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      return complex<float>(arr[t],arr[t+coffs]);
    }

    complex<float> get_value(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3] || i4<0 || i4>=dims[4]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3,i4).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4];  
      return complex<float>(arr[t],arr[t+coffs]);
    }


    void set(const Gindex& ix, complex<float> x) const{
      int t=ix(strides);  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set(const int i0, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set(const int i0, const int i1, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set(const int i0, const int i1, const int i2, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set(const int i0, const int i1, const int i2, const int i3, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>dims[3]) throw std::out_of_range("index "+Gindex(i0,i1,i2,i3).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set(const int i0, const int i1, const int i2, const int i3, const int i4, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=5 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>dims[3] || i4<0 || i4>dims[4]) throw std::out_of_range("index "+Gindex(i0,i1,i2,i3,i4).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }


    void set_value(const Gindex& ix, complex<float> x) const{
      int t=ix(strides);  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set_value(const int i0, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set_value(const int i0, const int i1, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set_value(const int i0, const int i1, const int i2, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set_value(const int i0, const int i1, const int i2, const int i3, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>dims[3]) throw std::out_of_range("index "+Gindex(i0,i1,i2,i3).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }

    void set_value(const int i0, const int i1, const int i2, const int i3, const int i4, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=5 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>dims[3] || i4<0 || i4>dims[4]) throw std::out_of_range("index "+Gindex(i0,i1,i2,i3,i4).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4];  
      arr[t]=std::real(x);
      arr[t+coffs]=std::imag(x);
    }


    void inc(const Gindex& ix, complex<float> x) const{
      int t=ix(strides);  
      arr[t]+=std::real(x);
      arr[t+coffs]+=std::imag(x);
    }

    void inc(const int i0, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      arr[t]+=std::real(x);
      arr[t+coffs]+=std::imag(x);
    }

    void inc(const int i0, const int i1, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      arr[t]+=std::real(x);
      arr[t+coffs]+=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, complex<float> x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]+=std::real(x);
      arr[t+coffs]+=std::imag(x);
    }


  public: // ---- Cells --------------------------------------------------------------------------------------

    /*
    CtensorB view_of_cell(const Gindex& cix){
      assert(coffs==1);
      assert(cix.size()<dims.size());
      CtensorB R(dims.chunk(cix.size()),fill_noalloc(),dev);
      R.arr=arr+cix(strides);
      R.is_view=true;
      return R;
    }

    const CtensorB view_of_cell(const Gindex& cix) const{
      assert(coffs==1);
      assert(cix.size()<dims.size());
      CtensorB R(dims.chunk(cix.size()),fill_noalloc(),dev);
      R.arr=arr+cix(strides);
      R.is_view=true;
      return R;
    }

    CtensorB get_cell(const Gindex& cix) const{
      return CtensorB(view_of_cell(cix),nowarn_flag());
    }
    */


  public: // ---- Chunks -------------------------------------------------------------------------------------


    void add_to_chunk(const int ix, const int offs, const CtensorB& x){
      /*
      assert(x.dev==dev);
      assert(k==x.k);
      for(int i=0; i<k; i++) 
	if(i!=ix) assert(dims[i]==x.dims[i]);
	else assert(dims[i]>=x.dims[i]);
      int subsize=x.asize;
      if(ix>0) subsize=x.strides[ix-1];
      int supsize=x.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*strides[ix];
	  //for(int m=0; m<x.dims[ix];; m++){
	  for(int i=0; i<subsize; i++){
	    arr[toffs+i]+=x.arr[j*subsize+i];
	    arrc[toffs+i]+=x.arrc[j*subsize+i];
	  }
	  //toffs+=strides[ix];
	  //}
	}
	return; 
      }
      */
      CNINE_UNIMPL();
      //const float alpha = 1.0;
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
    }


    void set_chunk(const int ix, const int offs, const CtensorB& x){
      /*
      assert(x.dev==dev);
      assert(k==x.k);
      for(int i=0; i<k; i++) 
	if(i!=ix) assert(dims[i]==x.dims[i]);
	else assert(dims[i]>=x.dims[i]);
      int subsize=x.asize;
      if(ix>0) subsize=x.strides[ix-1];
      int supsize=x.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*strides[ix];
	  //for(int m=0; m<x.dims[ix];; m++){
	  for(int i=0; i<subsize; i++){
	    arr[toffs+i]=x.arr[j*subsize+i];
	    arrc[toffs+i]=x.arrc[j*subsize+i];
	  }
	  //toffs+=strides[ix];
	  //}
	}
	return; 
      }
      */
      CNINE_UNIMPL();
      //const float alpha = 1.0;
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
    }


    void add_chunk_of(const CtensorB& x, const int ix, const int offs, const int n){
      /*
      assert(k==x.k);
      for(int i=0; i<k; i++) 
	if(i!=ix) assert(dims[i]==x.dims[i]);
	else assert(x.dims[i]>=dims[i]);
      int subsize=strides[ix];
      int supsize=x.asize/(strides[ix]*dims[ix]);
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];
      int jxstride=x.asize;
      if(ix>0) jxstride=x.strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  for(int m=0; m<n; m++){
	    for(int i=0; i<subsize; i++){
	      arr[j*jstride+m*strides[ix]+i]+=x.arr[j*jxstride+(m+offs)*x.strides[ix]+i];
	      arrc[j*jstride+m*strides[ix]+i]+=x.arrc[j*jxstride+(m+offs)*x.strides[ix]+i];
	    }
	  }
	}
	return; 
      }
      */
      CNINE_UNIMPL();
    }


    CtensorB chunk(const int ix, const int offs, const int n=1) const{
      Gdims _dims(dims);
      _dims[ix]=n;
      CtensorB x(_dims,fill::raw,dev);
      /*
      int subsize=strides[ix];
      int supsize=asize/(strides[ix]*dims[ix]);
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];
      int jxstride=x.asize;
      if(ix>0) jxstride=x.strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  for(int m=0; m<n; m++){
	    for(int i=0; i<subsize; i++){
	      x.arr[j*jxstride+m*x.strides[ix]+i]=arr[j*jstride+(m+offs)*strides[ix]+i];
	      x.arrc[j*jxstride+m*x.strides[ix]+i]=arrc[j*jstride+(m+offs)*strides[ix]+i];
	    }
	  }
	}
	return x; 
      }
      */
      CNINE_UNIMPL();
      return x; 
    }


  public: // ---- Slices -------------------------------------------------------------------------------------


    void add_to_slice(const int ix, const int offs, const CtensorB& x){
      /*
      assert(k==x.k+1);
      assert(x.dev==dev);
      for(int i=0; i<ix; i++) assert(dims[i]==x.dims[i]);
      for(int i=ix; i<x.k; i++) assert(dims[i+1]==x.dims[i]);
      int subsize=x.asize;
      if(ix>0) subsize=x.strides[ix-1];
      int supsize=x.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];

      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*strides[ix];
	  for(int i=0; i<subsize; i++){
	    arr[toffs+i]+=x.arr[j*subsize+i];
	    arrc[toffs+i]+=x.arrc[j*subsize+i];
	  }
	}
	return; 
      }
      */
      CNINE_UNIMPL();
    }


    void add_to_slices(const int ix, const vector<const CtensorB*> v){
      /*
      assert(v.size()==dims[ix]);
      const CtensorA& x=*v[0];
      assert(k==x.k+1);
      for(int i=0; i<ix; i++) assert(dims[i]==x.dims[i]);
      for(int i=ix; i<x.k; i++) assert(dims[i+1]==x.dims[i]);
      int subsize=x.asize;
      if(ix>0) subsize=x.strides[ix-1];
      int supsize=x.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];

      if(dev==0){
	for(int m=0; m<dims[ix]; m++){
	  for(int j=0; j<supsize; j++){
	    int toffs=j*jstride+m*strides[ix];
	    const CtensorA& x=*v[m];
	    for(int i=0; i<subsize; i++){
	      arr[toffs+i]+=x.arr[j*subsize+i];
	      arrc[toffs+i]+=x.arrc[j*subsize+i];
	    }
	  }
	}
	return; 
      }
      */
      CNINE_UNIMPL();
    }


    void add_slice_of(const CtensorB& x, const int ix, const int offs){
      /*
      assert(x.dev==dev);
      assert(x.k==k+1);
      for(int i=0; i<ix; i++) assert(dims[i]==x.dims[i]);
      for(int i=ix; i<k; i++) assert(x.dims[i+1]==dims[i]);
      int subsize=asize;
      if(ix>0) subsize=strides[ix-1];
      int supsize=asize/subsize;
      int jstride=x.asize; 
      if(ix>0) jstride=x.strides[ix-1];
      
      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*x.strides[ix];
	  for(int i=0; i<subsize; i++){
	    arr[j*subsize+i]+=x.arr[toffs+i];
	    arrc[j*subsize+i]+=x.arrc[toffs+i];
	  }
	}
	return; 
      }
      */
      CNINE_UNIMPL();
    }


    CtensorB slice(const int ix, const int offs) const{
      CtensorB R(dims.remove(ix),fill::raw,dev);
      /*
      int subsize=R.asize;
      if(ix>0) subsize=R.strides[ix-1];
      int supsize=R.asize/subsize;
      int jstride=asize; 
      if(ix>0) jstride=strides[ix-1];
      
      if(dev==0){
	for(int j=0; j<supsize; j++){
	  int toffs=j*jstride+offs*strides[ix];
	  for(int i=0; i<subsize; i++){
	    R.arr[j*subsize+i]=arr[toffs+i];
	    R.arrc[j*subsize+i]=arrc[toffs+i];
	  }
	}
	return R; 
      }
      */
      CNINE_UNIMPL();
      return R; 
    }


  public: // ---- CtensorB valued operations -----------------------------------------------------------------


    CtensorB conj() const{
      if(dev==0){
	CtensorB R(dims,fill::raw,0);
	int s=strides.back();
	for(int i=0; i<asize; i++) R.arr[i*s]=arr[i*s];
	for(int i=0; i<asize; i++) R.arr[i*s+coffs]=-arr[i*s+coffs];
	return R;
      }
      if(dev==1){
	CtensorB R(*this);
	const float alpha=-1.0;
	CUBLAS_SAFE(cublasSscal(cnine_cublas, asize, &alpha, R.arrg, 2));
	return R;
      }
      return *this;
    }

    CtensorB transp() const{
      CNINE_NDIMS_IS_2((*this));
      CtensorB R=CtensorB::raw(dims.transp(),dev);
      if(dev==0){
	for(int i=0; i<dims[0]; i++)
	  for(int j=0; j<dims[1]; j++)
	    R.set(j,i,(*this)(i,j));
      }
      if(dev==1){
	#ifdef _WITH_CUDA
	CtensorB R=CtensorB::raw(dims,dev);
	//const complex<float> alpha = 1.0;
	//const complex<float> beta = 0.0;
	//CUBLAS_SAFE(cublasCgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	//    &alpha,arrg,I,&beta,R.arrg,J,R.arrg,J));
	cuComplex alpha; 
	alpha.x=1.0;
	alpha.y=0.0;
	cuComplex beta;
	beta.x=0.0;
	beta.y=0.0;
	const int I=dims[0]; 
	const int J=dims[1]; 
	CUBLAS_SAFE(cublasCgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	    &alpha,reinterpret_cast<cuComplex*>(arrg),I,&beta,
	    reinterpret_cast<cuComplex*>(R.arrg),J,
	    reinterpret_cast<cuComplex*>(R.arrg),J));
	#endif 
	return R;
      }
      return *this;
    }

    CtensorB herm() const{
      CNINE_UNIMPL();
      return *this;
    }

    CtensorB plus(const CtensorB& y) const{
      CtensorB R(*this);
      R.add(y);
      return R;
    }
  
    CtensorB operator+(const CtensorB& y) const{
      CtensorB R(*this);
      R.add(y);
      return R;
    }

    CtensorB operator-(const CtensorB& y) const{
      CtensorB R(*this);
      R.subtract(y);
      return R;
    }

    CtensorB operator/(const CtensorB& y) const{ // TODO
      CNINE_CPUONLY();
      assert(y.memsize==memsize);
      CtensorB R=zeros_like(*this);
      for(int i=0; i<asize; i++){
	complex<float> t=complex<float>(arr[i*2],arr[i*2+coffs])/complex<float>(y.arr[i*2],y.arr[i*2+coffs]);
	R.arr[i*2]=std::real(t);
	R.arr[i*2+coffs]=std::imag(t);
      }
      for(int i=0; i<memsize; i++){
	R.arr[i]=arr[i]/y.arr[i];
      }
      return R;
    }


  public: // ---- Scalar valued operations ------------------------------------------------------------------

    
    complex<float> inp(const CtensorB& y) const{
      CNINE_DEVICE_SAME(y);
      CNINE_DIMS_EQ(dims,y.dims);
      CNINE_CPUONLY();
      complex<float> t=0;
      for(int i=0; i<asize; i++){
	t+=complex<float>(arr[2*i],arr[2*i+1])*complex<float>(y.arr[2*i],y.arr[2*i+1]);
      }
      return t;
    }

    float norm2() const{
      CNINE_CPUONLY();
      float t=0;
      for(int i=0; i<memsize; i++){
	t+=arr[i]*arr[i];
      }
      return t;
    }

    float diff2(const CtensorB& y) const{
      CNINE_DEVICE_SAME(y);
      CNINE_DIMS_EQ(dims,y.dims);
      CNINE_CPUONLY();
      float t=0;
      for(int i=0; i<memsize; i++){
	float d=arr[i]-y.arr[i];
	t+=d*d;
      }
      return t;
    }
  
  
public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const CtensorB& x){
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      if(dev==0){
	for(int i=0; i<memsize; i++) arr[i]+=x.arr[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, memsize, &alpha, x.arrg, 1, arrg, 1));
      }
    }

    void operator+=(const CtensorB& x){
      add(x);
    }
  
    void add(const CtensorB& x, const float c){
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      if(dev==0){
	for(int i=0; i<memsize; i++) arr[i]+=x.arr[i]*c;
	return;
      }
      if(dev==1){
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, memsize, &c, x.arrg, 1, arrg, 1));
      }
    }
  
  
    void add(const CtensorB& x, const complex<float> c){
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      float cr=std::real(c);
      float ci=std::imag(c);
      if(dev==0){
	//for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr-x.arrc[i]*ci;
	//for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*cr+x.arr[i]*ci;
	return;
      }
      if(dev==1){
	const float mci=-ci; 
	//CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
	//CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &mci, x.arrgc, 1, arrg, 1));
	//CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &ci, x.arrg, 1, arrgc, 1));
	//CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, x.arrgc, 1, arrgc, 1));
      }
    }


    void add_conj(const CtensorB& x){
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) 
	  arr[i*2]+=x.arr[i*2];
	for(int i=0; i<asize; i++) 
	  arr[i*2+1]-=x.arr[i*2+1];
	return; 
      }
      if(dev==1){
	CNINE_CPUONLY();
      }
    }

    void add_conj(const CtensorB& x, const float v){
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) 
	  arr[i*2]+=v*x.arr[i*2];
	for(int i=0; i<asize; i++) 
	  arr[i*2+1]-=v*x.arr[i*2+1];
	return; 
      }
      if(dev==1){
	CNINE_CPUONLY();
      }
    }

    void add_conj(const CtensorB& x, const complex<float> v){
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++){
	  complex<float> t=std::conj(complex<float>(x.arr[2*i],x.arr[2*i+1]))*v;
	  arr[i*2]+=std::real(t);
	  arr[i*2+1]+=std::imag(t);
	}
	return; 
      }
      if(dev==1){
	CNINE_CPUONLY();
      }
    }

    void add(const CtensorB& x, const RscalarA& c){
      add(x,c.val);
    }

    void add(const CtensorB& x, const CscalarA& c){
      add(x,c.val);
    }

    void add_cconj(const CtensorB& x, const CscalarA& c){
      add(x,std::conj(c.val));
    }

    void subtract(const CtensorB& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<memsize; i++) arr[i]-=x.arr[i];
	return; 
      }
      if(dev==1){
	const float alpha = -1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, memsize, &alpha, x.arrg, 1, arrg, 1));
      }
    }

    void operator-=(const CtensorB& x){
      subtract(x);
    }



    /*
    void add_matmul(const CtensorB& _x, const CtensorB& M, const int d){
      assert(M.getk()==2);
      assert(_x.getk()>d);
      assert(_x.dims(d)==M.dims(1));

      auto x=_x.like_matrix();
      auto r=this->like_matrix();

      int I=x.dims(1);
      int J=x.dims(0);
      for(int i=0; i<I; i++){
	for(int K=0; k<K; k++){
	  complex<float> t=0;
	  for(int j=0; j<J; j++){
	    //t+=x(i,j)*

	      //for(int i=0; i<r.dims(0); i++)
	  }
	}
      }
    }
    */

      
    void add_gather(const CtensorB& x, const Rmask1& mask){
      int k=dims.size();
      assert(x.dims.size()==k);
      if(k==2) Aggregator(view2(),x.view2(),mask);
      if(k==3) Aggregator(view3(),x.view3(),mask);
      if(k==4) Aggregator(view4(),x.view4(),mask);
    }


  public: // ---- Into Operations ----------------------------------------------------------------------------


    void add_norm2_into(CscalarA& r) const{
	r.val+=inp(*this);
    }

    void add_inp_into(CscalarA& r, const CtensorB& A) const{
      r.val+=inp(A);
    }


  public: // ---- Matrix multiplication -------------------------------------------------------------------
    

    void add_mprod_AA(const CtensorB& x, const CtensorB& y){
      add_Mprod_AA<0>(x,y);
    }


    // The last nx indices of x are contracted with the first ny indices of y
    // Selector: x is conjugated if selector is 1 or 3
    // Selector: y is conjugated if selector is 2 or 3

    template<int selector> 
    void add_Mprod_AA(const CtensorB& x, const CtensorB& y, const int nx=1, const int ny=1){
      CNINE_DEVICE_SAME(x);
      CNINE_DEVICE_SAME(y);
      CNINE_NDIMS_IS_2((*this));
      CNINE_NDIMS_IS_2(x);
      CNINE_NDIMS_IS_2(y);

      const int n0=dims[0];
      const int n1=dims[1];
      const int I=x.dims[1];
      assert(x.dims[0]==n0);
      assert(y.dims[1]==n1);
      assert(y.dims[0]==I);

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
	#ifdef _WITH_CUBLAS
	cuComplex alpha;
	alpha.x=1.0f;
	alpha.y=0.0f;
	#ifndef _OBJFILE
	CUBLAS_SAFE(cublasCgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,n1,n0,x.dims[1],&alpha,
	    reinterpret_cast<cuComplex*>(y.arr),y.dims[1], 
	    reinterpret_cast<cuComplex*>(x.arr),x.dims[1],&alpha,
	    reinterpret_cast<cuComplex*>(arr),n1)); 
	#endif
	#endif
      }
    }

    template<int selector> 
    void add_Mprod_AT(const CtensorB& x, const CtensorB& y, const int nx=1, const int ny=1){
      CNINE_UNIMPL();
    }

    template<int selector> 
    void add_Mprod_TA(const CtensorB& x, const CtensorB& y, const int nx=1, const int ny=1){
      CNINE_UNIMPL();
    }



  public: // ---- Special functions --------------------------------------------------------------------------


    void add_ReLU(const CtensorB& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.asize==asize);
      assert(x.dev==dev);
      for(int i=0; i<memsize; i++) arr[i]+=(x.arr[i]>0)*x.arr[i];
    }

    void add_ReLU(const CtensorB& x, const float alpha){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.asize==asize);
      assert(x.dev==dev);
      for(int i=0; i<memsize; i++) arr[i]+=((x.arr[i]>0)+alpha*(x.arr[i]<0))*x.arr[i];
    }


  public: // ---- Experimental -------------------------------------------------------------------------------


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    /*
    string str_as_array(const int k, const string indent="") const{
      ostringstream oss;
      assert(k<dims.size());
      Gdims arraydims=dims.chunk(0,k);
      arraydims.foreach_index([&](const vector<int>& ix){
	  oss<<indent<<"Cell"<<Gindex(ix)<<endl;
	  oss<<get_cell(ix).str(indent)<<endl;
	});
      return oss.str();
    }
    */

    string repr() const{
      return "<cnine::CtensorB"+dims.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const CtensorB& x){
      stream<<x.str(); return stream;}
   







  };

}


#endif 
    /*
    */
