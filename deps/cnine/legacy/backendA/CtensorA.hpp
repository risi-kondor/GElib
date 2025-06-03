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


#ifndef _CnineCtensorA
#define _CnineCtensorA

#include "Cnine_base.hpp"
#include "CnineObject.hpp"
#include "Gdims.hpp"
#include "Gtensor.hpp"
#include "CscalarA.hpp"
#include "CtensorA_accessor.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{


  //template<typename OBJ>
  //class Flock;


  class CtensorArrayA;


  class CtensorA: public CnineObject{ //, public CnineBackendObject{
  public:

    int k;
    Gdims dims;
    int nbu=-1;
    int dev=0;

    friend class CtensorArrayA;
    //friend class CtensorAflock;
    //friend class Flock<CtensorA>;

    //protected:

    vector<int> strides;
    int asize=0;
    int memsize=0;
    int cst=0; 

    bool is_view=false;

    float* arr=nullptr;
    float* arrc=nullptr;
    float* arrg=nullptr;
    float* arrgc=nullptr;

  public:

    //CtensorA(){}

    ~CtensorA(){
      if(!is_view && arr) {delete[] arr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
    }

    string classname() const{
      return "CtensorA";
    }

    string describe() const{
      return "CtensorA"+dims.str();
    }

    CtensorArrayA* array_type(){
      return reinterpret_cast<CtensorArrayA*>(this);
    }

    const CtensorArrayA* const_array_type(){
      return reinterpret_cast<const CtensorArrayA*>(this);
    }

    CtensorA():
      CtensorA(Gdims({0})){}


  public: // ---- Constructors -----------------------------------------------------------------------------

    
    CtensorA(const Gdims& _dims, const int _dev=0): 
      dims(_dims), dev(_dev), strides(_dims.size()){

      CNINE_CHECK_DEV(if(dev<0||dev>1) throw std::invalid_argument("Cnine error in CtensorA: device must be 0 or 1"));

      k=dims.size();
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      cst=roundup(asize,32); 
      memsize=2*cst; 

      if(dev==0){
	arr=new float[memsize];
	arrc=arr+cst; 
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst;
      }

    }


    CtensorA(const Gdims& _adims, const Gdims& _dims, const int _dev=0): // for CtensorArray
      dims(_adims,_dims), dev(_dev), strides(_adims.size()+_dims.size()){

      k=dims.size();
      const int ak=_adims.size();
      const int dk=_dims.size();

      strides[k-1]=1;
      for(int i=dk-2; i>=0; i--)
	strides[ak+i]=strides[ak+i+1]*_dims[i+1];
      const int cellstride=roundup(strides[ak]*_dims[0],32); 

      strides[ak-1]=cellstride;
      for(int i=ak-2; i>=0; i--)
	strides[i]=strides[i+1]*_adims[i+1];
      asize=strides[0]*_adims[0];

      cst=roundup(asize,32); 
      memsize=2*cst; 

      if(dev==0){
	arr=new float[memsize];
	arrc=arr+cst; 
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst;
      }

    }


    CtensorA(const Gdims& _adims, const Gdims& _dims, const fill_noalloc& dummy, const int _dev=0): // for CtensorArray
      dims(_adims,_dims), dev(_dev), strides(_adims.size()+_dims.size()){

      k=dims.size();
      const int ak=_adims.size();
      const int dk=_dims.size();

      strides[k-1]=1;
      for(int i=dk-2; i>=0; i--)
	strides[ak+i]=strides[ak+i+1]*_dims[i+1];
      const int cellstride=roundup(strides[ak]*_dims[0],32); 

      strides[ak-1]=cellstride;
      for(int i=ak-2; i>=0; i--)
	strides[i]=strides[i+1]*_adims[i+1];
      asize=strides[0]*_adims[0];

      cst=roundup(asize,32); 
      memsize=2*cst; 
    }


    CtensorA(const int _k, const Gdims& _dims, const vector<int>_strides, const int _asize, 
      const int _memsize, const int _cst, const int _dev=0):
      k(_k), dims(_dims), dev(_dev), strides(_strides), asize(_asize), memsize(_memsize), cst(_cst){

      if(dev==0){
	arr=new float[memsize];
	arrc=arr+cst;
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst;
      }

    }

    /*
    CtensorA(const int _k, const Gdims& _dims, const int _nbu, const vector<int>& _strides, const int _asize, 
      const int _memsize, const int _cst, const int _dev, const float* _arr, const float* _arrc, const view_flag& flag):
      k(_k), dims(_dims), nbu(_nbu), strides(_strides), asize(_asize), memsize(_memsize), cst(_cst), dev(_dev), 
      arr(_arr), arrc(_arrc), is_view(true){}
    */
    
    CtensorA(const int _k, const Gdims& _dims, const int _nbu, const vector<int>& _strides, const int _asize, 
      const int _memsize, const int _dev, float* _arr, float* _arrc, const view_flag& flag):
      k(_k), dims(_dims), nbu(_nbu), dev(_dev), strides(_strides), asize(_asize), memsize(_memsize), cst(_asize), 
      is_view(true), arr(_arr), arrc(_arrc){
    }
    

  public: // ---- Shape and strides -------------------------------------------------------------------------

    
    void reshape(const Gdims& _dims){
      assert(_dims.asize()==asize);
      dims=_dims;
      k=dims.size();
      strides.resize(k);
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      cst=roundup(asize,32); 
      memsize=2*cst; 
    }

    /*
    void reshape(const Gdims& _dims, const Gdims& _adims):
      dims(_adims,_dims), strides(_adims.size()+_dims.size()){

	k=dims.size();
	const int ak=_adims.size();
	const int dk=_dims.size();

      strides[k-1]=1;
      for(int i=dk-2; i>=0; i--)
	strides[ak+i]=strides[ak+i+1]*_dims[i+1];
      const int cellstride=roundup(strides[ak]*_dims[0],32); 

      strides[ak-1]=cellstride;
      for(int i=ak-2; i>=0; i--)
	strides[i]=strides[i+1]*_adims[i+1];
      asize=strides[0]*_adims[0];

      cst=roundup(asize,32); 
      memsize=2*cst; 
      }
    */


  public: // ---- Filled constructors -----------------------------------------------------------------------


    CtensorA(const Gdims& _dims, const int _nbu, const int _dev):
      CtensorA(_dims.prepend(_nbu),_dev){
      //if(_nbu>=0) bundle=true;
    }

    CtensorA(const Gdims& _dims, const fill_raw& dummy, const int _dev=0): 
      CtensorA(_dims,_dev){}
    
    CtensorA(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      CtensorA(_dims,_dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    CtensorA(const Gdims& _dims, const fill_ones& dummy, const int _dev=0): 
      CtensorA(_dims,fill::raw,0){
      std::fill(arr,arr+asize,1);
      std::fill(arrc,arrc+asize,0);
      if(_dev==1) move_to_device(_dev);
    }

    CtensorA(const Gdims& _dims, const fill_identity& dummy, const int _dev=0): 
      CtensorA(_dims,fill::raw,0){
      assert(dims[k-1]==dims[k-2]);
      std::fill(arr,arr+memsize,0);
      for(int i=0; i<dims[k-1]; i++)
	arr[i*(strides[k-2]+1)]=1;
      move_to_device(_dev);
    }

    CtensorA(const Gdims& _dims, const fill_gaussian& dummy, const int _dev):
      CtensorA(_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=distr(rndGen);
      for(int i=0; i<asize; i++) arrc[i]=distr(rndGen);
      move_to_device(_dev);
    }

    CtensorA(const Gdims& _dims, const fill_gaussian& dummy, const float c, const int _dev):
      CtensorA(_dims,fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=c*distr(rndGen);
      for(int i=0; i<asize; i++) arrc[i]=c*distr(rndGen);
      move_to_device(_dev);
    }

    CtensorA(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      CtensorA(_dims,fill::zero,0){
      for(int i=0; i<asize; i++) arr[i]=i;
      move_to_device(_dev);
    }
	  
    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorA(const Gdims& _dims, const int _nbu, const FILLTYPE& fill, const int _dev=0):
      CtensorA(_dims.prepend(_nbu),fill,_dev){
      nbu=_nbu;
      //if(_nbu>=0) bundle=true;
    }
	  
    CtensorA(const Gdims& _dims, const int _nbu, const fill_gaussian& fill, const float c, const int _dev=0):
      CtensorA(_dims.prepend(_nbu),fill,c,_dev){
      nbu=_nbu;
      //if(_nbu>=0) bundle=true;
    }


  public: // ---- Lambda constructors -------------------------------------------------------------------------


    CtensorA(const Gdims& _dims, std::function<complex<float>(const int i, const int j)> fn):
      CtensorA(_dims,fill::raw,0){
      assert(dims.size()==2);
      for(int i=0; i<dims[0]; i++)
	for(int j=0; j<dims[1]; j++)
	  set_value(i,j,fn(i,j));
    }

    CtensorA(const CtensorA& x, std::function<complex<float>(const complex<float>)> fn):
      CtensorA(x.dims,fill::raw,0){
      assert(x.dev==0);
      for(int i=0; i<asize; i++){
	complex<float> t=fn(complex<float>(x.arr[i],x.arrc[i]));
	arr[i]=std::real(t);
	arrc[i]=std::imag(t);
      }
    }

    CtensorA(const CtensorA& x, std::function<complex<float>(const int i, const int j, const complex<float>)> fn):
      CtensorA(x.dims,fill::raw,0){
      assert(x.dev==0);
      assert(x.dims.size()==2);
      for(int i=0; i<dims[0]; i++)
	for(int j=0; j<dims[1]; j++)
	  set_value(i,j,fn(i,j,complex<float>(x(i,j))));
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    CtensorA(const CtensorA& x): 
      CtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.dev){
      if(dev==0){
	std::copy(x.arr,x.arr+asize,arr);
	std::copy(x.arrc,x.arrc+asize,arrc);
      }
#ifdef _WITH_CUDA
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));
	CUDA_SAFE(cudaMemcpy(arrgc,x.arrgc,asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
#endif 
    }
        
    CtensorA(const CtensorA& x, const nowarn_flag& dummy): 
      CtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.dev){
      if(dev==0){
	std::copy(x.arr,x.arr+asize,arr);
	std::copy(x.arrc,x.arrc+asize,arrc);
      }
#ifdef _WITH_CUDA
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));
	CUDA_SAFE(cudaMemcpy(arrgc,x.arrgc,asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
#endif 
    }
        
    CtensorA(const CtensorA& x, const int _dev): 
      CtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,_dev){
      if(dev==0){
	if(x.dev==0){
	  std::copy(x.arr,x.arr+asize,arr);
	  std::copy(x.arrc,x.arrc+asize,arrc);
	}
	if(x.dev==1){
	  CUDA_SAFE(cudaMemcpy(arr,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToHost)); 
	  CUDA_SAFE(cudaMemcpy(arrc,x.arrgc,asize*sizeof(float),cudaMemcpyDeviceToHost)); 
	}
      }
      if(dev==1){
#ifdef _WITH_CUDA
	if(x.dev==0){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arr,asize*sizeof(float),cudaMemcpyHostToDevice));
	  CUDA_SAFE(cudaMemcpy(arrgc,x.arrc,asize*sizeof(float),cudaMemcpyHostToDevice));
	}
	if(x.dev==1){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaMemcpy(arrgc,x.arrgc,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
	}
#endif 
      }
    }

    CtensorA(const CtensorA& x, const view_flag& dummy){
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; dev=x.dev; 
      memsize=x.memsize; cst=x.cst; 
      arr=x.arr; arrc=x.arrc;
      arrg=x.arrg; arrgc=x.arrgc;
      is_view=true;
      //cout<<"CtensorA view "<<endl;
    }
        
    CtensorA(CtensorA&& x){ //: CtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.dev){
      k=x.k; dims=x.dims; nbu=x.nbu; 
      strides=x.strides; asize=x.asize; dev=x.dev; 
      memsize=x.memsize; cst=x.cst; 
      arr=x.arr; x.arr=nullptr; 
      arrc=x.arrc; x.arrc=nullptr;  
      arrg=x.arrg; x.arrg=nullptr;
      arrgc=x.arrgc; x.arrgc=nullptr;
      is_view=x.is_view;
      //cout<<"move CtensorA "<<endl; 
    }

    CtensorA(const CtensorA& x, const fill_raw& dummy): 
      CtensorA(x.k,x.dims,x.strides,x.asize,x.memsize,x.cst,x.dev){}

    CtensorA* clone() const{
      return new CtensorA(*this);
    }

    CtensorA& operator=(const CtensorA& x){
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; dev=x.dev;
      memsize=x.memsize; cst=x.cst;

      if(is_view){
	if(dev==0){
	  std::copy(x.arr,x.arr+asize,arr);
	  std::copy(x.arrc,x.arrc+asize,arrc);
	}
	if(dev==1){
#ifdef _WITH_CUDA
	  CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaMemcpy(arrgc,x.arrgc,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
#endif 
	}
	return *this;
      }

      delete arr;
#ifdef _WITH_CUDA
      if(arrg){CUDA_SAFE(cudaFree(arrg));}
#endif
      if(dev==0){
	arr=new float[memsize]; 
	arrc=arr+cst; 
	std::copy(x.arr,x.arr+asize,arr);
	std::copy(x.arrc,x.arrc+asize,arrc);
      }
      if(dev==1){
#ifdef _WITH_CUDA
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst;
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
	CUDA_SAFE(cudaMemcpy(arrgc,x.arrgc,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
#endif 
      }
      
      return *this;
    }


    CtensorA& operator=(CtensorA&& x){
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; dev=x.dev; 
      memsize=x.memsize; cst=x.cst; 
      if(!is_view && arr) delete arr;
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
      arr=x.arr; x.arr=nullptr; 
      arrc=x.arr; x.arrc=nullptr; 
      arrg=x.arrg; x.arrg=nullptr; 
      arrgc=x.arrgc; x.arrgc=nullptr; 
      is_view=x.is_view;
      return *this;
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    CtensorA(const Gtensor<complex<float> >& x, const int _dev=0): 
      CtensorA(x.dims,fill::raw){
      assert(dev==0);
      for(int i=0; i<asize; i++){
	arr[i]=std::real(x.arr[i]);
	arrc[i]=std::imag(x.arr[i]);
      }
      move_to_device(_dev);
    }
    
    Gtensor<complex<float> > gtensor() const{
      if(dev>0) return CtensorA(*this,0).gtensor();
      Gtensor<complex<float> > R(dims,fill::raw);
      assert(dev==0);
      for(int i=0; i<asize; i++){
	R.arr[i]=complex<float>(arr[i],arrc[i]);
      }
      return R;
    }


#ifdef _WITH_ATEN

 static bool is_viewable(const at::Tensor& T){
    if(T.dim()>0 && T.size(0)==2 && T.stride(0)%32==0) return true;
    else return false;
  }

  CtensorA(const at::Tensor& T){
      CNINE_CONVERT_FROM_ATEN_WARNING();
      assert(typeid(T.type().scalarType())==typeid(float));

      T.contiguous();
      k=T.dim()-1;
      if(k<=0 || T.size(0)!=2) throw std::out_of_range("CtensorA: first dimension of tensor must be 2, corresponding to the real and imaginary parts.");
      dims=Gdims(k,fill_raw());
      for(int i=0; i<k ; i++){
	dims[i]=T.size(i+1);
      }
      strides.resize(k);
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      cst=roundup(asize,32); 
      memsize=2*cst; 

      dev=T.type().is_cuda();
      if(dev==0){
	arr=new float[memsize];
	arrc=arr+cst; 
	std::copy(T.data<float>(),T.data<float>()+asize,arr);
	std::copy(T.data<float>()+asize,T.data<float>()+2*asize,arrc);
      }

      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst; 
	CUDA_SAFE(cudaMemcpy(arrg,T.data<float>(),asize*sizeof(float),cudaMemcpyDeviceToDevice));
	CUDA_SAFE(cudaMemcpy(arrgc,T.data<float>()+asize,asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
    }


    static CtensorA view(at::Tensor& T){
      T.contiguous();
      
      CtensorA R;
      R.k=T.dim()-1;
      R.dims.resize(R.k);
      for(int i=0; i<R.k ; i++)
	R.dims[i]=T.size(i+1);
      R.strides.resize(R.k);
      for(int i=0; i<R.k; i++)
	R.strides[i]=T.stride(i+1);
      R.asize=R.strides[0]*R.dims[0]; // TODO
      R.cst=R.asize; 
      R.memsize=2*R.cst; 
      R.dev=T.type().is_cuda();
      R.is_view=true;

      if(R.dev==0){
	R.arr=T.data<float>();
	R.arrc=T.data<float>()+R.cst;
      }
      
      if(R.dev==1){
	R.arrg=T.data<float>();
	R.arrgc=T.data<float>()+R.cst;
      }

      return R;
    }
    

    at::Tensor torch() const{
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

#endif 


  public: // ---- Transport -----------------------------------------------------------------------------------


    CtensorA& move_to_device(const int _dev){

      if(_dev==0){
	if(dev==0) return *this;
 	delete[] arr;
	arr=new float[memsize];
	arrc=arr+cst;
	CUDA_SAFE(cudaMemcpy(arr,arrg,asize*sizeof(float),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaMemcpy(arrc,arrgc,asize*sizeof(float),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arrg));
	const_cast<CtensorA*>(this)->arrg=nullptr;
	dev=0;
	return *this;
      }

      if(_dev>0){
	if(dev==_dev) return *this;
	if(arrg) CUDA_SAFE(cudaFree(arrg));
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
	arrgc=arrg+cst;
	CUDA_SAFE(cudaMemcpy(arrg,arr,asize*sizeof(float),cudaMemcpyHostToDevice));  
	CUDA_SAFE(cudaMemcpy(arrgc,arrc,asize*sizeof(float),cudaMemcpyHostToDevice));  
	delete[] arr;
	const_cast<CtensorA*>(this)->arr=nullptr;
	dev=_dev;
	return *this;
      }
      
      return *this;
    }
    
    CtensorA& move_to(const struct device& _dev){
      return move_to_device(_dev.id());
    }
    
    CtensorA to(const struct device& _dev) const{
      return CtensorA(*this,_dev.id());
    }

    CtensorA to_device(const int _dev) const{
      return CtensorA(*this,_dev);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nbu() const{
      return nbu;
    }

    Gdims get_dims() const{
      return dims;
    }

    int get_dim(const int i) const{
      return dims[i];
    }

    int get_dev() const{
      return dev;
    }

    int get_device() const{
      return dev;
    }

    CtensorA_accessor accessor(){
      return CtensorA_accessor(arr,arrc,strides);
    }

    int dim(const int i) const{
      return dims[i];
    }

    int combined_size(const int a, const int b) const{
      assert(b<=k);
      assert(a<=b);
      if(b>0 && strides[b-1]==0) return 0;
      if(a>0) return (strides[a-1])/(strides[b-1]);
      if(b>0) return asize/strides[b-1];
      return 1; 
    }


  public: // ---- Gindex case ---------


    complex<float> operator()(const Gindex& ix) const{
      CNINE_ASSERT(dev==0,"CtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return complex<float>(arr[t],arrc[t]);
    }

    complex<float> get_value(const Gindex& ix) const{
      CNINE_ASSERT(dev==0,"CtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return complex<float>(arr[t],arrc[t]);
    }
    
    void set_value(const Gindex& ix, const complex<float>& v){
      CNINE_ASSERT(dev==0,"CtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]=std::real(v);
      arrc[t]=std::imag(v);
    }

    void inc(const Gindex& ix, const complex<float>& v){
      CNINE_ASSERT(dev==0,"CtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(ix.check_range(dims));
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]+=std::real(v);
      arrc[t]+=std::imag(v);
    }

    CscalarA get(const Gindex& ix) const{
      CNINE_ASSERT(dev==0,"CtensorA::get(...) not implemented for GPU.\n");
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return CscalarA(complex<float>(arr[t],arrc[t]));
    }
    
    void set(const Gindex& ix, const CscalarA& x){
      CNINE_ASSERT(dev==0,"CtensorA::get(...) not implemented for GPU.\n");
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]=std::real(x.val);
      arrc[t]=std::imag(x.val);
    }

    complex<float> get_value_at(const int t) const{
      CNINE_ASSERT(dev==0,"CtensorA::get_value_at() not implemented for GPU.\n");
      return complex<float>(arr[t],arrc[t]);
    }
    
    void set_value_at(const int t, const complex<float>& v){
      CNINE_ASSERT(dev==0,"CtensorA::set_value_at() not implemented for GPU.\n");
      arr[t]=std::real(v);
      arrc[t]=std::imag(v);
    }


  public: // ---- k=1 special cases ---- 


    complex<float> operator()(const int i0) const{
      CNINE_ASSERT(dev==0,"CtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      assert(k==1);
      int t=i0*strides[0];
      return complex<float>(arr[t],arrc[t]);
    }

    complex<float> get_value(const int i0) const{
      CNINE_ASSERT(dev==0,"CtensorA::get not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      assert(k==1);
      int t=i0*strides[0];
      return complex<float>(arr[t],arrc[t]);
    }

    void set_value(const int i0, const complex<float> x){
      CNINE_ASSERT(dev==0,"CtensorA::set not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      assert(k==1);
      int t=i0*strides[0];
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const complex<float> x){
      CNINE_ASSERT(dev==0,"CtensorA::inc not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }

    CscalarA get(const int i0) const{
      CNINE_ASSERT(dev==0,"CtensorA::get(int) not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      return CscalarA(complex<float>(arr[t],arrc[t]));
    }

    void set(const int i0, const CscalarA& x){
      CNINE_ASSERT(dev==0,"CtensorA::set not implemented for GPU.\n");
      assert(k==1);
      int t=i0*strides[0];
      arr[t]=std::real(x.val);
      arrc[t]=std::imag(x.val);
    }


  public: // ---- k=2 special cases ---- 


    complex<float> operator()(const int i0, const int i1) const{
      CNINE_ASSERT(dev==0,"CtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      return complex<float>(arr[t],arrc[t]);
    }

    complex<float> get_value(const int i0, const int i1) const{
      CNINE_ASSERT(dev==0,"CtensorA::get not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      return complex<float>(arr[t],arrc[t]);
    }

    void set_value(const int i0, const int i1, const complex<float> x){
      CNINE_ASSERT(dev==0,"CtensorA::set not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const complex<float> x){
      CNINE_ASSERT(dev==0,"CtensorA::inc not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }

    CscalarA get(const int i0, const int i1) const{
      CNINE_ASSERT(dev==0,"CtensorA::get not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      return CscalarA(complex<float>(arr[t],arrc[t]));
    }

    void set(const int i0, const int i1, const CscalarA& x){
      CNINE_ASSERT(dev==0,"CtensorA::set not implemented for GPU.\n");
      assert(k==2);
      int t=i0*strides[0]+i1*strides[1];
      arr[t]=std::real(x.val);
      arrc[t]=std::imag(x.val);
    }


  public: // ---- k=3 special cases ----


    complex<float> operator()(const int i0, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0, "CtensorA::operator() not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return complex<float>(arr[t],arrc[t]);
    }

    complex<float> get_value(const int i0, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0, "CtensorA::get not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return complex<float>(arr[t],arrc[t]);
    }

    void set_value(const int i0, const int i1, const int i2, const complex<float> x){
      CNINE_ASSERT(dev==0, "CtensorA::set not implemented for GPU.\n");
      CNINE_CHECK_RANGE(if(k!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]=std::real(x);
      arrc[t]=std::imag(x);
    }

    void inc(const int i0, const int i1, const int i2, const complex<float> x){
      CNINE_ASSERT(dev==0, "CtensorA::inc not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]+=std::real(x);
      arrc[t]+=std::imag(x);
    }

    CscalarA get(const int i0, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0, "CtensorA::get not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return CscalarA(complex<float>(arr[t],arrc[t]));
    }

    void set(const int i0, const int i1, const int i2, const CscalarA& x){
      CNINE_ASSERT(dev==0, "CtensorA::set not implemented for GPU.\n");
      assert(k==3);
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]=std::real(x.val);
      arrc[t]=std::imag(x.val);
    }


    // ---- Cumulative 


    void add_element_into(CscalarA& r, const Gindex& ix){
      if(nbu==-1){
	r.val+=get_value(ix);
      }else{
	CNINE_UNIMPL();
      }
    }

    void add_to_element(const Gindex& ix, CscalarA& r){
      if(nbu==-1){
	inc(ix,r.val);
      }else{
	CNINE_UNIMPL();
      }
    }


  public: // ---- Chunks -------------------------------------------------------------------------------------


    void add_to_chunk(const int ix, const int offs, const CtensorA& x){
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
      CNINE_UNIMPL();
      //const float alpha = 1.0;
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
    }


    void set_chunk(const int ix, const int offs, const CtensorA& x){
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
      CNINE_UNIMPL();
      //const float alpha = 1.0;
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
    }


    void add_chunk_of(const CtensorA& x, const int ix, const int offs, const int n){
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
      CNINE_UNIMPL();
    }


    CtensorA chunk(const int ix, const int offs, const int n=1) const{
      Gdims _dims(dims);
      _dims[ix]=n;
      CtensorA x(_dims,fill::raw,dev);
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
      CNINE_UNIMPL();
      return x; 
    }


  public: // ---- Slices -------------------------------------------------------------------------------------


    void add_to_slice(const int ix, const int offs, const CtensorA& x){
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
      CNINE_UNIMPL();
    }


    void add_to_slices(const int ix, const vector<const CtensorA*> v){
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
      CNINE_UNIMPL();
    }


    void add_slice_of(const CtensorA& x, const int ix, const int offs){
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
      CNINE_UNIMPL();
    }


    CtensorA slice(const int ix, const int offs) const{
      CtensorA R(dims.remove(ix),fill::raw,dev);
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
      CNINE_UNIMPL();
      return R; 
    }


  public: // ---- In-place Operations ------------------------------------------------------------------------


    void set_zero(){
      if(dev==0){
	std::fill(arr,arr+asize,0);
	std::fill(arrc,arrc+asize,0);
      }
#ifdef _WITH_CUDA
      if(dev==1){
	CUDA_SAFE(cudaMemset(arrg,0,asize*sizeof(float)));
	CUDA_SAFE(cudaMemset(arrgc,0,asize*sizeof(float)));
      }
#endif
    }


    void inplace_times(const complex<float> c){
      if(dev==0){
	for(int i=0; i<asize; i++){
	  complex<float> t=c*complex<float>(arr[i],arrc[i]);
	  arr[i]=std::real(t);
	  arrc[i]=std::imag(t);
	}
	return; 
      }
      if(dev==1){
	IFCUDA(
	       const float cr = std::real(c);
	       const float ci = std::imag(c);
	       const float mci = -std::imag(c);
	       CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, arrg, 1, arrg, 1));
	       CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &mci, arrgc, 1, arrg, 1));
	       CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, arrgc, 1, arrgc, 1));
	       CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &ci, arrg, 1, arrgc, 1));
	)
      }
    }

    void inplace_times(const CscalarA& c){
      inplace_times(c.val);
    }

    void inplace_div(const CscalarA& c){
      inplace_times(complex<float>(1.0)/c.val);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    CtensorA conj() const{
      if(dev==0){
	CtensorA R(dims,fill::raw,0);
	std::copy(arr,arr+asize,R.arr);
	for(int i=0; i<asize; i++) R.arrc[i]=-arrc[i];
	return R;
      }
      CtensorA R(dims,fill::zero,dev);
      IFCUDA(
	     const float alpha = 1.0;
	     const float malpha = -1.0;
	     CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, arrg, 1, R.arrg, 1));
	     CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &malpha, arrgc, 1, R.arrgc, 1));
	     )
      return R;
    }

    CtensorA* conjp() const{
      if(dev==0){
	CtensorA* R=new CtensorA(dims,fill::raw,0);
	std::copy(arr,arr+asize,R->arr);
	for(int i=0; i<asize; i++) R->arrc[i]=-arrc[i];
	return R;
      }
      CtensorA* R=new CtensorA(dims,fill::zero,dev);
      IFCUDA(
	     const float alpha = 1.0;
	     const float malpha = -1.0;
	     CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, arrg, 1, R->arrg, 1));
	     CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &malpha, arrgc, 1, R->arrgc, 1));
	     )
      return R;
    }

    CtensorA transp(const int n=1) const{
      const int J=combined_size(0,n);
      const int I=asize/J;
      if(dev==0){
	CtensorA R({I,J},fill::raw,0);
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]=arr[j*I+i];
	    R.arrc[i*J+j]=arrc[j*I+i];
	  }
	return R;
      }
      CtensorA R(dims,fill::zero,dev);
      const float alpha = 1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrg,I,&beta,R.arrg,J,R.arrg,J));
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrgc,I,&beta,R.arrgc,J,R.arrgc,J));
      return R;
    }

    CtensorA* transpp(const int n=1) const{
      const int J=combined_size(0,n);
      const int I=asize/J;
      if(dev==0){
	CtensorA* R=new CtensorA({I,J},fill::raw,0);
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    R->arr[i*J+j]=arr[j*I+i];
	    R->arrc[i*J+j]=arrc[j*I+i];
	  }
	return R;
      }
      CtensorA* R=new CtensorA(dims,fill::zero,dev);
      const float alpha = 1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrg,I,&beta,R->arrg,J,R->arrg,J));
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrgc,I,&beta,R->arrgc,J,R->arrgc,J));
      return R;
    }

    CtensorA herm(const int n=1) const{
      const int J=combined_size(0,n);
      const int I=asize/J;
      if(dev==0){
	CtensorA R({I,J},fill::raw,0);
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]=arr[j*I+i];
	    R.arrc[i*J+j]=-arrc[j*I+i];
	  }
	return R;
      }
      CtensorA R(dims,fill::zero,dev);
      const float alpha = 1.0;
      const float malpha = -1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrg,I,&beta,R.arrg,J,R.arrg,J));
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &malpha,arrgc,I,&beta,R.arrgc,J,R.arrgc,J));
      return R;
    }

    CtensorA* hermp(const int n=1) const{
      const int J=combined_size(0,n);
      const int I=asize/J;
      if(dev==0){
	CtensorA* R=new CtensorA({I,J},fill::raw,0);
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    R->arr[i*J+j]=arr[j*I+i];
	    R->arrc[i*J+j]=-arrc[j*I+i];
	  }
	return R;
      }
      CtensorA* R=new CtensorA(dims,fill::zero,dev);
      const float alpha = 1.0;
      const float malpha = -1.0;
      const float beta = 0.0;
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &alpha,arrg,I,&beta,R->arrg,J,R->arrg,J));
      CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	  &malpha,arrgc,I,&beta,R->arrgc,J,R->arrgc,J));
      return R;
    }

    CtensorA plus(const CtensorA& x) const{
      CtensorA R(*this);
      R.add(x);
      return R;
    }


    /*
    CtensorA* conj() const{
      return new CtensorA(CFtensor::conj());
    }

    CtensorA* transp() const{
      return new CtensorA(CFtensor::transp());
    }

    CtensorA* herm() const{
      return new CtensorA(CFtensor::herm());
    }
    */
    
    
    CtensorA* divide_colsp(const CtensorA& N) const{
      return new CtensorA(divide_cols(N));
    }

    CtensorA* normalize_colsp() const{
      return new CtensorA(normalize_cols());
    }

    CtensorA divide_cols(const CtensorA& N) const{
      assert(N.dev==dev);
      assert(k>=2);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      CtensorA R(dims,fill::zero,0);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float z=N.arr[a*J+j];
	    for(int i=0; i<I; i++){
	      R.arr[offs+i*J+j]=arr[offs+i*J+j]/z;
	      R.arrc[offs+i*J+j]=arrc[offs+i*J+j]/z;
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
      return R;
    }

    CtensorA normalize_cols() const{
      Gdims ndims=dims.chunk(0,dims.size()-1);
      const int J=dims[dims.size()-1];
      const int I=asize/J;
      CtensorA R(*this);
      if(dev==0){
	for(int i=0; i<I; i++){
	  float tr=0;
	  float ti=0;
	  for(int j=0; j<J; j++){
	    tr+=R.arr[i*J+j]*R.arr[i*J+j];
	    ti+=R.arrc[i*J+j]*R.arrc[i*J+j];
	  }
	  float z=sqrt(tr+ti);
	  for(int j=0; j<J; j++){
	    R.arr[i*J+j]/=z;
	    R.arrc[i*J+j]/=z;
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
      return R;
    }

    
    float norm2() const{
      if(dev==0){
      float t=0; 
      for(int i=0; i<asize; i++) 
	t+=arr[i]*arr[i]-arrc[i]*arrc[i]; 
      return t;
      }
      float tr=0;
      float ti=0;
#ifdef _WITH_CUBLAS
      cublasSdot(cnine_cublas, asize, arrg, 1, arrg, 1, &tr);
      cublasSdot(cnine_cublas, asize, arrgc, 1, arrgc, 1, &ti);
#else
      CNINE_NOCUDA_ERROR;
#endif       
      return tr+ti;
    }


    complex<float> inp(const CtensorA& x) const{
      assert(asize==x.asize);
      if(asize==0) return 0; 
      assert(x.dev==dev);
      if(dev==0){
	float tr=0; 
	float ti=0; 
	for(int i=0; i<asize; i++){
	  tr+=arr[i]*x.arr[i]+arrc[i]*x.arrc[i];
	  ti+=arrc[i]*x.arr[i]-arr[i]*x.arrc[i];
	}
	//{CoutLock lk; cout<<*this<<endl<<endl; cout<<"  "<<asize<<" "<<tr<<":"<<ti<<endl;}
	return complex<float>(tr,ti);
      }
      float a=0;
      float b=0; 
      float c=0;
      float d=0; 
#ifdef _WITH_CUBLAS
      cudaDeviceSynchronize();
      cublasSdot(cnine_cublas, asize, arrg, 1, x.arrg, 1, &a);
      cublasSdot(cnine_cublas, asize, arrgc, 1, x.arrgc, 1, &b);
      cublasSdot(cnine_cublas, asize, arrgc, 1, x.arrg, 1, &c);
      cublasSdot(cnine_cublas, asize, arrg, 1, x.arrgc, 1, &d);
      cudaDeviceSynchronize();
#else
      CNINE_NOCUDA_ERROR;
#endif       
      return complex<float>(a+b,c-d);
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void set(const CtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	std::copy(x.arr,x.arr+asize,arr);
	std::copy(x.arrc,x.arrc+asize,arrc);
	return; 
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
	CUDA_SAFE(cudaMemcpy(arrgc,x.arrgc,asize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }


    void add(const CtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
      }
    }


    void add(const CtensorA& x, const float c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*c;
	return;
      }
      if(dev==1){
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &c, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &c, x.arrgc, 1, arrgc, 1));
      }
    }


    void add(const CtensorA& x, const complex<float> c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      float cr=std::real(c);
      float ci=std::imag(c);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr-x.arrc[i]*ci;
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*cr+x.arr[i]*ci;
	return;
      }
      if(dev==1){
	const float mci=-ci; 
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &mci, x.arrgc, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &ci, x.arrg, 1, arrgc, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, x.arrgc, 1, arrgc, 1));
      }
    }

    void add(const CtensorA& x, const RscalarA& c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(c.nbu==-1);
      add(x,c.val);
    }

    void add(const CtensorA& x, const CscalarA& c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(c.nbu==-1);
      add(x,c.val);
    }

    void add_cconj(const CtensorA& x, const CscalarA& c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(c.nbu==-1);
      add(x,std::conj(c.val));
    }

    void add_conj(const CtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
	for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	const float malpha = -1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &malpha, x.arrgc, 1, arrgc, 1));
      }
    }

    void add_conj(const CtensorA& x, complex<float> c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(asize==x.asize);
      assert(x.dev==dev);
      float cr=std::real(c);
      float ci=-std::imag(c);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*cr-x.arrc[i]*ci;
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]*cr+x.arr[i]*ci;
	return;
      }
      if(dev==1){
	const float mci=-ci; 
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &ci, x.arrgc, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &mci, x.arrg, 1, arrgc, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &cr, x.arrgc, 1, arrgc, 1));
      }
    }

    void add_conj(const CtensorA& x, const CscalarA& c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(c.nbu==-1);
      add_conj(x,c.val);
    }

    void add_prod(const RscalarA& c, const CtensorA& A){
      CNINE_CHECK_SIZE(dims.check_eq(A.dims));
      assert(c.nbu==-1);
      add(A,c.val);
    }
 
    void add_prod(const CscalarA& c, const CtensorA& A){
      CNINE_CHECK_SIZE(dims.check_eq(A.dims));
      assert(c.nbu==-1);
      add(A,c.val);
    }
    
    void add_prod_cconj(const CscalarA& c, const CtensorA& A){
      CNINE_CHECK_SIZE(dims.check_eq(A.dims));
      assert(c.nbu==-1);
      add(A,std::conj(c.val));
    }
 
    void add_prod_c_times_conj(const CscalarA& c, const CtensorA& A){
      CNINE_CHECK_SIZE(dims.check_eq(A.dims));
      assert(c.nbu==-1);
      add_conj(A,c.val);
    }

    void add_divide(const CtensorA& x, const float c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      add(x,1.0/c);
    }

    void add_divide(const CtensorA& x, const complex<float> c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      add(x,complex<float>(1.0)/c);
    }

    void add_divide(const CtensorA& x, const RscalarA& c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(c.nbu==-1);
      add(x,1.0/c.val);
    }

    void add_divide(const CtensorA& x, const CscalarA& c){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(c.nbu==-1);
      add(x,complex<float>(1.0)/c.val);
    }


    void add_sum(const vector<CtensorA*> v){
      const int N=v.size();
      if(N==0) return; 
      if(dev==0){
	for(int i=0; i<N; i++){
	  const CtensorA& o=*v[i];
	  assert(o.asize==asize);
	  assert(o.dev==dev);
	  for(int j=0; j<asize; j++){
	    arr[j]+=o.arr[j];
	    arrc[j]+=o.arrc[j];
	  }
	}
	return;
      }
      const float alpha = 1.0;
      for(int i=0; i<N; i++){
	const CtensorA& o=*v[i];
	assert(o.asize==asize);
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, o.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, o.arrgc, 1, arrgc, 1));
	//cudaDeviceSynchronize();
      }
    }

    void subtract(const CtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.dev==dev);
      assert(asize==x.asize);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]-=x.arr[i];
	for(int i=0; i<asize; i++) arrc[i]-=x.arrc[i];
	return;
      }
      const float c=-1.0; 
      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &c, x.arrg, 1, arrg, 1));
      CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &c, x.arrgc, 1, arrgc, 1));
    }

    void add_plus(const CtensorA& x, const CtensorA& y){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      CNINE_CHECK_SIZE(dims.check_eq(y.dims));
      assert(asize==x.asize);
      assert(asize==y.asize);
      assert(x.dev==dev);
      assert(y.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]+y.arr[i];
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]+y.arrc[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, y.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, y.arrgc, 1, arrgc, 1));
      }
    }

    void add_minus(const CtensorA& x, const CtensorA& y){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      CNINE_CHECK_SIZE(dims.check_eq(y.dims));
      assert(asize==x.asize);
      assert(asize==y.asize);
      assert(x.dev==dev);
      assert(y.dev==dev);
      if(dev==0){
	for(int i=0; i<asize; i++) arr[i]+=x.arr[i]-y.arr[i];
	for(int i=0; i<asize; i++) arrc[i]+=x.arrc[i]-y.arrc[i];
	return; 
      }
      if(dev==1){
	const float alpha = 1.0;
	const float malpha = -1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrgc, 1, arrgc, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &malpha, y.arrg, 1, arrg, 1));
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &malpha, y.arrgc, 1, arrgc, 1));
      }
    }

    void add_transp(const CtensorA& x, const int n=1) const{
      CNINE_CHECK_SIZE(dims.check_eq(x.dims.transpose()));
      assert(asize==x.asize);
      assert(x.dev==dev);
      const int J=x.combined_size(0,n);
      const int I=x.asize/J;
      if(dev==0){
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    arr[i*J+j]+=x.arr[j*I+i];
	    arrc[i*J+j]+=x.arrc[j*I+i];
	  }
	return;
      }
      if(dev==1){
	const float alpha = 1.0;
	const float beta = 1.0;
	CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	    &alpha,x.arrg,I,&beta,arrg,J,arrg,J));
	CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	    &alpha,x.arrgc,I,&beta,arrgc,J,arrgc,J));
      }
    }

    void add_herm(const CtensorA& x, const int n=1) const{
      CNINE_CHECK_SIZE(dims.check_eq(x.dims.transpose()));
      assert(asize==x.asize);
      assert(x.dev==dev);
      const int J=x.combined_size(0,n);
      const int I=x.asize/J;
      if(dev==0){
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    arr[i*J+j]+=x.arr[j*I+i];
	    arrc[i*J+j]-=x.arrc[j*I+i];
	  }
	return;
      }
      if(dev==1){
	const float alpha = 1.0;
	const float malpha = -1.0;
	const float beta = 1.0;
	CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	    &alpha,x.arrg,I,&beta,arrg,J,arrg,J));
	CUBLAS_SAFE(cublasSgeam(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,
	    &malpha,x.arrgc,I,&beta,arrgc,J,arrgc,J));
      }
    }


  public: // ---- Into Operations ----------------------------------------------------------------------------


    void add_norm2_ito(CscalarA& r) const{
      if(nbu==-1){
	r.val+=inp(*this);
      }else{
	CNINE_UNIMPL();
      }
    }

    void add_norm2_into(CscalarA& r) const{
      if(nbu==-1){
	r.val+=inp(*this);
      }else{
	CNINE_UNIMPL();
      }
    }

    void add_inp_into(CscalarA& r, const CtensorA& A) const{
      CNINE_CHECK_SIZE(dims.check_eq(A.dims));
      if(nbu==-1){
	r.val+=inp(A);
      }else{
	CNINE_UNIMPL();
      }
    }


  public: // ---- Normalization ------------------------------------------------------------------------------


    void add_col_norms(const CtensorA& x){
      assert(x.dev==dev);
      int xk=x.dims.size();
      assert(xk>=2);
      const int J=x.dims[xk-1];
      const int I=x.dims[xk-2];
      const int A=x.asize/(I*J);
      assert(asize==A*J);

      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float t=0;
	    for(int i=0; i<I; i++){
	      t+=x.arr[offs+i*J+j]*x.arr[offs+i*J+j]+x.arrc[offs+i*J+j]*x.arrc[offs+i*J+j];
	    }
	    arr[a*J+j]+=sqrt(t);
	  }
	}
	return;
      }
      CNINE_UNIMPL();
    }


    void add_col_norms_back(const CtensorA& G, const CtensorA& X, const CtensorA& N){
      assert(G.dev==dev);
      assert(X.dev==dev);
      assert(N.dev==dev);
      assert(k>=2);
      assert(X.asize==asize);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      assert(G.asize==N.asize);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float z=G.arr[a*J+j]/N.arr[a*J+j];
	    for(int i=0; i<I; i++){
	      arr[offs+i*J+j]+=X.arr[offs+i*J+j]*z;
	      arrc[offs+i*J+j]+=X.arrc[offs+i*J+j]*z;
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
    }


    void add_divide_cols(const CtensorA& X, const CtensorA& N){
      assert(X.dev==dev);
      assert(N.dev==dev);
      assert(k>=2);
      assert(X.asize==asize);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    //complex<float> z=complex<float>(N.arr[a*J+j],N.arrc[a*J+j]);
	    float z=N.arr[a*J+j];
	    for(int i=0; i<I; i++){
	      //complex<float> u=complex<float>()
	      arr[offs+i*J+j]+=X.arr[offs+i*J+j]/z;
	      arrc[offs+i*J+j]+=X.arrc[offs+i*J+j]/z;
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
    }


    void add_divide_cols_back0(const CtensorA& G, const CtensorA& N){
      assert(G.dev==dev);
      assert(N.dev==dev);
      assert(k>=2);
      const int J=dims[k-1];
      const int I=dims[k-2];
      const int A=asize/(I*J);
      assert(N.asize==asize/I);
      assert(G.asize==asize);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    complex<float> n(N.arr[a*J+j],0); //-N.arrc[a*J+j]);
	    //complex<float> z=complex<float>(1,0)/n/n; //complex<float>(G.arr[a*J+j],G.arrc[a*J+j])/n/n;
	    for(int i=0; i<I; i++){
	      complex<float> u=complex<float>(G.arr[offs+i*J+j],G.arrc[offs+i*J+j])/n;
	      //complex<float> u=z*complex<float>(G.arr[offs+i*J+j],G.arrc[offs+i*J+j])*
	      //complex<float>(X.arr[offs+i*J+j],-X.arrc[offs+i*J+j]);
	      arr[offs+i*J+j]+=std::real(u);
	      arrc[offs+i*J+j]+=std::imag(u);
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
    }


    void add_divide_cols_back1(const CtensorA& G, const CtensorA& X, const CtensorA& N){
      assert(G.dev==dev);
      assert(X.dev==dev);
      assert(N.dev==dev);
      const int _k=G.k;
      assert(_k>=2);
      assert(G.dims==X.dims);
      assert(dims==N.dims);
      const int J=G.dims[_k-1];
      const int I=G.dims[_k-2];
      const int A=G.asize/(I*J);
      assert(N.asize==G.asize/I);
      assert(asize==N.asize);
      if(dev==0){
	for(int a=0; a<A; a++){
	  int offs=a*I*J;
	  for(int j=0; j<J; j++){
	    float z=-pow(N.arr[a*J+j],-2);
	    for(int i=0; i<I; i++){ // improve
	      complex<float> t=complex<float>(G.arr[offs+i*J+j],G.arrc[offs+i*J+j])*
		complex<float>(X.arr[offs+i*J+j],-X.arrc[offs+i*J+j])*z;
	      arr[a*J+j]+=std::real(t);
	      arrc[a*J+j]+=std::imag(t);
	    }
	  }    
	}
      }else{
	CNINE_UNIMPL(); 
      }
    }

  protected: // ---- Matrix multiplication -------------------------------------------------------------------
  public:
    
    // The last nx indices of x are contracted with the first ny indices of y
    // Selector: x is conjugated if selector is 1 or 3
    // Selector: y is conjugated if selector is 2 or 3

    template<int selector> 
    void add_Mprod_AA(const CtensorA& x, const CtensorA& y, const int nx=1, const int ny=1){

      if(x.asize==0 || y.asize==0) return;

      const int K=x.combined_size(x.k-nx,x.k);
      assert(y.combined_size(0,ny)==K);

      const int I=x.combined_size(0,x.k-nx);
      const int J=y.combined_size(ny,y.k);
      assert(asize==I*J);
      if(asize==0) return;

      if(dev==0){
	assert(x.dev==0);
	assert(y.dev==0);

	const int istridex=K;
	const int istrider=J;
	const int pstridey=J;

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    float tr=0; 
	    float ti=0;
	    for(int p=0; p<K; p++){
	      int qx=i*istridex+p;
	      int qy=p*pstridey+j;
	      float xr=x.arr[qx];
	      float xi=x.arrc[qx];
	      float yr=y.arr[qy];
	      float yi=y.arrc[qy];
	      if (selector==0) {tr+=xr*yr-xi*yi; ti+=xr*yi+xi*yr;}
	      if (selector==1) {tr+=xr*yr+xi*yi; ti+=xr*yi-xi*yr;}
	      if (selector==2) {tr+=xr*yr+xi*yi; ti+=(-xr*yi)+xi*yr;}
	      if (selector==3) {tr+=xr*yr-xi*yi; ti-=xr*yi+xi*yr;}
	    }
	    int qr=i*istrider+j;
	    arr[qr]+=tr;
	    arrc[qr]+=ti;
	  }
      }

      if(dev>0){

#ifdef _WITH_CUDA
	float alpha0=1.0;
	float alpha1=1.0;
	float alpha2=1.0;
	float alpha3=1.0;
	float beta=1.0;
	
	if (selector==0||selector==3) alpha1=-1.0;
	if (selector==2||selector==3) alpha2=-1.0;
	if (selector==1||selector==3) alpha3=-1.0;
#endif 
    
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha0,
	    y.arrg,J,x.arrg,K,&beta,arrg,J)); 
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha1,
	    y.arrgc,J,x.arrgc,K,&beta,arrg,J)); 
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha2,
	    y.arrgc,J,x.arrg,K,&beta,arrgc,J)); 
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha3,
	    y.arrg,J,x.arrgc,K,&beta,arrgc,J)); 
	//cudaDeviceSynchronize(); 
      }

    }


    // The last nx indices of x are contracted with the last ny indices of y

    template<int selector> 
    void add_Mprod_AT(const CtensorA& x, const CtensorA& y, const int nx=1, const int ny=1){

      if(x.asize==0 || y.asize==0) return;

      const int K=x.combined_size(x.k-nx,x.k);
      assert(y.combined_size(y.k-ny,y.k)==K);

      const int I=x.combined_size(0,x.k-nx);
      const int J=y.combined_size(0,y.k-ny);
      assert(asize==I*J);
      if(asize==0) return;

      if(dev==0){
	assert(x.dev==0);
	assert(y.dev==0);

	const int istridex=K;
	const int istrider=J;
	const int jstridey=K;

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    float tr=0; 
	    float ti=0;
	    for(int p=0; p<K; p++){
	      int qx=i*istridex+p;
	      int qy=p+j*jstridey;
	      float xr=x.arr[qx];
	      float xi=x.arrc[qx];
	      float yr=y.arr[qy];
	      float yi=y.arrc[qy];
	      if (selector==0) {tr+=xr*yr-xi*yi; ti+=xr*yi+xi*yr;}
	      if (selector==1) {tr+=xr*yr+xi*yi; ti+=xr*yi-xi*yr;}
	      if (selector==2) {tr+=xr*yr+xi*yi; ti+=(-xr*yi)+xi*yr;}
	      if (selector==3) {tr+=xr*yr-xi*yi; ti-=xr*yi+xi*yr;}
	    }
	    int qr=i*istrider+j;
	    arr[qr]+=tr;
	    arrc[qr]+=ti;
	  }

      }

      if(dev>0){

#ifdef _WITH_CUDA
	float alpha0=1.0;
	float alpha1=1.0;
	float alpha2=1.0;
	float alpha3=1.0;
	float beta=1.0;
	
	if (selector==0||selector==3) alpha1=-1.0;
	if (selector==2||selector==3) alpha2=-1.0;
	if (selector==1||selector==3) alpha3=-1.0;
#endif 

	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha0,
	    y.arrg,K,x.arrg,K,&beta,arrg,J)); 
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha1,
	    y.arrgc,K,x.arrgc,K,&beta,arrg,J)); 
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha2,
	    y.arrgc,K,x.arrg,K,&beta,arrgc,J)); 
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha3,
	    y.arrg,K,x.arrgc,K,&beta,arrgc,J)); 
	//cudaDeviceSynchronize(); 
      }

    }


    // The first nx indices of x are contracted with the first ny indices of y

    template<int selector>
    void add_Mprod_TA(const CtensorA& x, const CtensorA& y, const int nx=1, const int ny=1){
  
      if(x.asize==0 || y.asize==0) return;

      const int K=x.combined_size(0,nx);
  
      assert(y.combined_size(0,ny)==K);

      const int I=x.combined_size(nx,x.k);
      const int J=y.combined_size(ny,y.k);
      assert(asize==I*J);
      if(asize==0) return;

      if(dev==0){
	assert(x.dev==0);
	assert(y.dev==0);

	const int istrider=J;
	const int pstridex=I;
	const int pstridey=J;

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    float tr=0; 
	    float ti=0;
	    for(int p=0; p<K; p++){
	      int qx=i+p*pstridex;
	      int qy=p*pstridey+j;
	      float xr=x.arr[qx];
	      float xi=x.arrc[qx];
	      float yr=y.arr[qy];
	      float yi=y.arrc[qy];
	      if (selector==0) {tr+=xr*yr-xi*yi; ti+=xr*yi+xi*yr;}
	      if (selector==1) {tr+=xr*yr+xi*yi; ti+=xr*yi-xi*yr;}
	      if (selector==2) {tr+=xr*yr+xi*yi; ti+=(-xr*yi)+xi*yr;}
	      if (selector==3) {tr+=xr*yr-xi*yi; ti-=xr*yi+xi*yr;}
	    }
	    int qr=i*istrider+j;
	    arr[qr]+=tr;
	    arrc[qr]+=ti;
	  }

      }

      if(dev>0){
	
#ifdef _WITH_CUDA
	float alpha0=1.0;
	float alpha1=1.0;
	float alpha2=1.0;
	float alpha3=1.0;
	float beta=1.0;
	
	if (selector==0||selector==3) alpha1=-1.0;
	if (selector==2||selector==3) alpha2=-1.0;
	if (selector==1||selector==3) alpha3=-1.0;
#endif 

	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,J,I,K,&alpha0,
	    y.arrg,J,x.arrg,I,&beta,arrg,J)); 
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,J,I,K,&alpha1,
	    y.arrgc,J,x.arrgc,I,&beta,arrg,J)); 
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,J,I,K,&alpha2,
	    y.arrgc,J,x.arrg,I,&beta,arrgc,J)); 
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_T,J,I,K,&alpha3,
	    y.arrg,J,x.arrgc,I,&beta,arrgc,J)); 
	//cudaDeviceSynchronize(); 
      }
      
    }

  public: // ---- Special functions --------------------------------------------------------------------------


    void add_ReLU(const CtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.asize==asize);
      assert(x.dev==dev);
      for(int i=0; i<asize; i++) arr[i]+=(x.arr[i]>0)*x.arr[i];
      for(int i=0; i<asize; i++) arrc[i]+=(x.arrc[i]>0)*x.arrc[i];
    }

    void add_ReLU(const CtensorA& x, const float alpha){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      assert(x.asize==asize);
      assert(x.dev==dev);
      for(int i=0; i<asize; i++) arr[i]+=((x.arr[i]>0)+alpha*(x.arr[i]<0))*x.arr[i];
      for(int i=0; i<asize; i++) arrc[i]+=((x.arrc[i]>0)+alpha*(x.arrc[i]<0))*x.arrc[i];
    }

    void add_ReLU_back(const CtensorA& g, const CtensorA& x){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      CNINE_CHECK_SIZE(dims.check_eq(g.dims));
      assert(x.asize==asize);
      assert(g.asize==asize);
      assert(x.dev==dev);
      assert(g.dev==dev);
      for(int i=0; i<asize; i++) arr[i]+=(x.arr[i]>0)*g.arr[i];
      for(int i=0; i<asize; i++) arrc[i]+=(x.arrc[i]>0)*g.arr[i];
    }

    void add_ReLU_back(const CtensorA& g, const CtensorA& x, const float alpha){
      CNINE_CHECK_SIZE(dims.check_eq(x.dims));
      CNINE_CHECK_SIZE(dims.check_eq(g.dims));
      assert(x.asize==asize);
      assert(g.asize==asize);
      assert(x.dev==dev);
      assert(g.dev==dev);
      for(int i=0; i<asize; i++) arr[i]+=((x.arr[i]>0)+(x.arr[i]<=0)*alpha)*g.arr[i];
      for(int i=0; i<asize; i++) arrc[i]+=((x.arrc[i]>0)+(x.arrc[i]<=0)*alpha)*g.arrc[i];
    }

    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    friend ostream& operator<<(ostream& stream, const CtensorA& x){
      stream<<x.str(); return stream;}
   
  };


  
  /*
  inline CtensorA& asCtensorA(Cobject* x, const char* s){
    return downcast<CtensorA>(x,s);
  }

  inline CtensorA& asCtensorA(Cnode* x, const char* s){
    return downcast<CtensorA>(x,s);
  }
  
  inline CtensorA& asCtensorA(Cnode& x, const char* s){
    return downcast<CtensorA>(x,s);
  }
  */


  //#define CTENSORB(x) asCtensorA(x,__PRETTY_FUNCTION__) 


}

#endif

    /*
    CtensorB(const string filename, const device_id& dev=0):
      CFtensor(filename,dev){}

    int save(const string filename) const{
      CFtensor::save(filename);
      return 0;
    }

    CtensorB(Bifstream& ifs): 
      CFtensor(ifs){
    }

    void serialize(Bofstream& ofs){
      CFtensor::serialize(ofs);
    }
    */

  //inline CtensorB& asCtensorB(Cobject* x){
  //return downcast<CtensorB>(x,"");
  //}

  //inline CtensorB& asCtensorB(Cnode* x){
  //return downcast<CtensorB>(x,"");
  //}

  //inline CtensorB& asCtensorB(Cnode& x){
  //return downcast<CtensorB>(x,"");
  //}

    /*
    CtensorB(const Gdims& _dims, const int _nbu, std::function<complex<float>(const int i, const int j)> fn, const int dev=0):
      CFtensor(_dims.prepend(_nbu),fill::raw), dims(_dims), nbu(_nbu){
      if(nbu==-1){
	for(int i=0; i<dims[0]; i++)
	  for(int j=0; j<dims[1]; j++)
	    CFtensor::set(i,j,fn(i,j));
      }else{
	for(int b=0; b<nbu; b++)
	  for(int i=0; i<dims[0]; i++)
	    for(int j=0; j<dims[1]; j++)
	      CFtensor::set(b,i,j);
      }
      if(dev>0) to_device(dev);
      CTENSORB_CREATE();
    }
	  
    CtensorB(const CtensorB& x, std::function<complex<float>(const complex<float>)> fn):
      CFtensor(x,fn), dims(x.dims){
      CTENSORB_CREATE();
    }

    CtensorB(const CtensorB& x, std::function<complex<float>(const int i, const int j, const complex<float>)> fn):
      CFtensor(x,fill::raw), dims(x.dims){
      assert(dims.size()==2);
      if(nbu==-1){
	for(int i=0; i<dims[0]; i++)
	  for(int j=0; j<dims[1]; j++)
	    CFtensor::set(i,j,fn(i,j,x.CFtensor::get(i,j)));
      }else{
	for(int b=0; b<nbu; b++)
	  for(int i=0; i<dims[0]; i++)
	    for(int j=0; j<dims[1]; j++)
	      CFtensor::set(b,i,j,fn(i,j,x.CFtensor::get(b,i,j)));
      }
      CTENSORB_CREATE();
    }
    */

    /*
    void mix_into(CtensorB& r, const CtensorB& x) const{
      to_device(0);
      assert(dims.size()==2);
      if(r.nbu==-1){
	assert(dims[0]==1);
	if(x.nbu==-1){
	  assert(dims[1]==1);
	  r.add(x,complex<float>(arr[0],arrc[0]));
	  return; 
	}else{
	  assert(dims[1]==x.nbu);
	  FCG_UNIMPL();
	  return;
	}
      }else{
	FCG_UNIMPL();
      }
    }
    */
    /*
    void mix_into(CscalarB& r, const CscalarB& x) const{
      to_device(0);
      assert(dims.size()==2);
      if(r.nbu==-1){
	assert(dims[0]==1);
	if(x.nbu==-1){
	  assert(dims[1]==1);
	  r.val+=complex<float>(arr[0],arrc[0])*x.val;
	  return; 
	}else{
	  assert(dims[1]==x.nbu);
	  for(int i=0; i<x.nbu; i++)
	    r.val+=complex<float>(arr[i],arrc[i])*x.arr[i];
	  return;
	}
      }else{
	assert(dims[0]==r.nbu);
	if(x.nbu==-1){
	  assert(dims[1]==1);
	  for(int i=0; i<r.nbu; i++)
	    r.arr[i]+=complex<float>(arr[i],arrc[i])*x.val;
	}else{
	  assert(dims[1]==x.nbu);
	  for(int i=0; i<r.nbu; i++){
	    complex<float> t=r.arr[i];
	    for(int j=0; j<x.nbu; j++)
	      t+=complex<float>(arr[i*x.nbu+j],arrc[i*x.nbu+j])*x.val;
	    r.arr[i]=t;
	  }
	}
      }
    }
    */
