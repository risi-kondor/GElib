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


#ifndef _CnineGtensor
#define _CnineGtensor

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
#endif 

#include <fstream>

#include "Cnine_base.hpp"
#include "Gindex.hpp"
#include "Gdims.hpp"

//#include "Bifstream.hpp"
//#include "Bofstream.hpp"


extern default_random_engine rndGen;


namespace cnine{

  extern string base_indent;


  template<class TYPE>
  class Gtensor{
  public:

    int k;
    int ak=0; 
    Gdims dims;
    vector<int> strides;
    int asize;
    int memsize=0;

    mutable TYPE* arr=nullptr;
    mutable TYPE* arrg=nullptr;

    bool is_view=false;
    bool is_contiguous=true;

    mutable int dev=0;

    ~Gtensor(){
      if(!is_view && arr) delete arr;
#ifdef _WITH_CUDA
      if(!is_view && arrg) {cout<<"deleting"<<endl; CUDA_SAFE(cudaFree(arrg));}
#endif
    }

    string classname() const {return "Cengine::Gtensor";}


  public: // ---- Constructors -------------------------------------------------------------------------------


    Gtensor(){
      arr=nullptr;
    }

    Gtensor(const Gdims& _dims, const struct device& _dev=0): 
      dims(_dims), strides(_dims.size()){
      make_strides();
      if(_dev.id()==0){
	arr=new TYPE[asize];
	dev=0;
      }
      if(_dev.id()==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, asize*sizeof(TYPE)));
	dev=1;
      }
    }

    Gtensor(const int _k, const Gdims& _dims, const vector<int>_strides, const int _asize, const int _device=0):
      k(_k), dims(_dims), strides(_strides), asize(_asize), dev(_device){
      memsize=asize;
      if(dev==0) arr=new TYPE[memsize];
#ifdef _WITH_CUDA
      if(dev==1) CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(TYPE)));
#endif
    }

    Gtensor(const int _k, const Gdims& _dims, const vector<int>_strides, const int _asize, const struct device& _dev=0):
      k(_k), dims(_dims), strides(_strides), asize(_asize), dev(_dev.id()){
      memsize=asize;
      if(dev==0) arr=new TYPE[memsize];
#ifdef _WITH_CUDA
      if(dev==1) CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(TYPE)));
#endif
    }

    Gtensor(const Gdims& _adims, const Gdims& _dims, const struct device& _dev=0): 
      k(_dims.size()+_adims.size()), dims(_adims,_dims), strides(_dims.size()){
      ak=_adims.size(); 
      make_strides(ak);
       if(_dev.id()==0){
	arr=new TYPE[memsize];
	dev=0;
      }
      if(_dev.id()==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(TYPE)));
	dev=1;
      }
    }

    Gtensor(const vector<TYPE>& v, const struct device& _dev=0):
      k(1), strides({1}), asize(v.size()){
      dims.push_back(asize);
      memsize=asize;
      arr=new TYPE[asize];
      for(int i=0; i<asize; i++){
	arr[i]=v[i];}
      to_device(_dev);
    }

    Gtensor(const vector<const Gtensor<TYPE>*> v){
      const int n=v.size(); 
      assert(n>0);
      const Gdims& sdims=v[0]->dims;
      const int smemsize=v[0]->smemsize;
      k=sdims.size()+1;
      dims=Gdims(n,sdims);
      make_strides();
      reallocate();
      for(int i=0; i<n; i++){
	assert(v[i]->dev==0);
	assert(v[i]->dims==sdims);
	std::copy(v[i]->arr, v[i]->arr+smemsize,arr+i*smemsize);
      }
    }


  private:

      
    void make_strides(){
      k=dims.size();
      strides.resize(k);
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      memsize=strides[0]*dims[0];
      asize=memsize; 
    }

    void make_strides(const int _ak){
      k=dims.size();
      strides.resize(k);
      strides[k-1]=1;
      for(int i=k-2; i>=ak; i--)
	strides[i]=strides[i+1]*dims[i+1];
      if(ak>0) strides[ak-1]=roundup(strides[ak]*dims[ak],32);
      for(int i=ak-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      memsize=strides[0]*dims[0];
      asize=dims.asize();
    }

    void reallocate(const struct device& _dev=0) const{
      if(_dev.id()==0){
	arr=new TYPE[memsize];
	dev=0;}
      if(_dev.id()==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(TYPE)));
	dev=1;}
    }


  public: // ---- Filled constructors -------------------------------------------------------------------------


    Gtensor(const Gdims& _dims, const fill_raw& dummy, const struct device& _dev=0): 
      Gtensor(_dims,_dev){}

    Gtensor(const Gdims& _dims, const fill_zero& dummy, const struct device& _dev=0): 
      Gtensor(_dims,_dev) {
      if(_dev.id()==0) std::fill(arr,arr+memsize,0);
      if(_dev.id()==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(TYPE)));
    }

    Gtensor(const Gdims& _dims, const fill_identity& dummy, const struct device& dev=0): 
      Gtensor(_dims) {
      assert(dims.size()==2);
      assert(dims[0]==dims[1]);
      std::fill(arr,arr+memsize,0);
      for(int i=0; i<dims[1]; i++)
	arr[i*(dims[0]+1)]=1;
      to_device(dev);
    }

    Gtensor(const Gdims& _dims, const fill_gaussian& dummy, const struct device& _dev=0):
      Gtensor(_dims){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=distr(rndGen);
      to_device(_dev);
    }

    Gtensor(const Gdims& _dims, const fill_cgaussian& dummy, const struct device& _dev=0):
      Gtensor(_dims){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=TYPE(distr(rndGen),distr(rndGen));
      to_device(_dev);
    }

    template<typename U=TYPE>
    Gtensor(const Gdims& _dims, const fill_gaussian& dummy, const struct device& _dev=0, 
	      typename std::enable_if<is_same<U,complex<float> >::value | is_same<U,complex<double> >::value >::type=0): 
      Gtensor(_dims){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=TYPE(distr(rndGen),distr(rndGen));
      to_device(_dev);
    }

    Gtensor(const Gdims& _dims, const fill_sequential& dummy, const struct device& _dev=0):
      Gtensor(_dims){
      for(int i=0; i<asize; i++) arr[i]=i;
      to_device(_dev);
    }

    //Gtensor(const Gdims& _dims, const fill_const<TYPE>& dummy, const device& _dev=0):
    //Gtensor(_dims){
    //for(int i=0; i<asize; i++) arr[i]=dummy.p;
    //to_device(_dev);
    //}

    Gtensor(const Gdims& _adims, const Gdims& _dims, const fill_zero& dummy, const struct device& _dev=0):
      Gtensor(_adims,_dims,_dev){
      normal_distribution<double> distr;
      if(_dev.id()==0) std::fill(arr,arr+memsize,0);
      if(_dev.id()==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(TYPE)));
    }

    Gtensor(const Gdims& _adims, const Gdims& _dims, const fill_raw& dummy, const struct device& _dev=0):
      Gtensor(_adims,_dims,_dev){
    }

    Gtensor(const Gdims& _adims, const Gdims& _dims, const fill_cgaussian& dummy, const struct device& _dev=0):
      Gtensor(_adims,_dims){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=TYPE(distr(rndGen),distr(rndGen));
      to_device(_dev);
    }

    Gtensor(const Gdims& _adims, const Gdims& _dims, const fill_gaussian& dummy, const struct device& _dev=0):
      Gtensor(_adims,_dims){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=distr(rndGen);
      to_device(_dev);
    }

    template<typename U=TYPE>
    Gtensor(const Gdims& _adims, const Gdims& _dims, const fill_gaussian& dummy, const struct device& _dev=0, 
	      typename std::enable_if<is_same<U,complex<float> >::value | is_same<U,complex<double> >::value >::type=0): 
      Gtensor(_adims,_dims){
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=TYPE(distr(rndGen),distr(rndGen));
      to_device(_dev);
    }


  public: // ---- Copying -------------------------------------------------------------------------------------


    Gtensor(const Gtensor<TYPE>& x): 
      Gtensor(x.k,x.dims,x.strides,x.asize,x.dev){
      ak=x.ak;
      memsize=x.memsize;
      //COPY_WARNING;
      if(dev==0) std::copy(x.arr,x.arr+memsize,arr);
      if(dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
    }
        
    Gtensor(const Gtensor<TYPE>& x, const nowarn_flag& dummy): 
      Gtensor(x.k,x.dims,x.strides,x.asize,x.dev){
      ak=x.ak;
      memsize=x.memsize;
      if(dev==0) std::copy(x.arr,x.arr+memsize,arr);
      if(dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
    }

    Gtensor(const Gtensor<TYPE>& x, const struct device& _dev): 
      Gtensor(x.k,x.dims,x.strides,x.asize,_dev){
      ak=x.ak;
      memsize=x.memsize;
      if(dev==0){
	if(x.dev==0) std::copy(x.arr,x.arr+memsize,arr);
#ifdef _WITH_CUDA
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToHost)); 
#endif
      }
      if(dev==1){
#ifdef _WITH_CUDA
	if(x.dev==0) CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(TYPE),cudaMemcpyHostToDevice));
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
#endif 
      }
    }
    
    Gtensor(Gtensor<TYPE>&& x){
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize;
      ak=x.ak; memsize=x.memsize;
      arr=x.arr; x.arr=nullptr; arrg=x.arrg; x.arrg=nullptr; 
      is_view=x.is_view;
      is_contiguous=x.is_contiguous;
      dev=x.dev;
    }
    
    Gtensor& operator=(const Gtensor<TYPE>& x){
      ak=x.ak;
      memsize=x.memsize;
      if(!is_view) delete arr;
      if(!is_view && arrg) cudaFree(arrg); 
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; dev=x.dev;
      if(dev==0){
	arr=new TYPE[memsize]; 
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
      return *this;
    }

    Gtensor& operator=(Gtensor<TYPE>&& x){
      ak=x.ak;
      memsize=x.memsize;
      if(!is_view && arr) delete arr;
      if(!is_view && arrg) CUDA_SAFE(cudaFree(arrg));
      k=x.k; dims=x.dims; strides=x.strides; asize=x.asize; dev=x.dev; 
      arr=x.arr; x.arr=nullptr; arrg=x.arrg; x.arrg=nullptr;
      is_view=x.is_view;
      return *this;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    Gtensor(const Gtensor<complex<TYPE> >& x): 
      Gtensor(x.k,x.dims,x.strides,x.asize,0){
      x.to_device(0);
      for(int i=0; i<asize; i++) arr[i]=std::real(x.arr[i]);
    }
        

    template<typename TYPE2>
    Gtensor(const Gtensor<TYPE2>& x): 
      Gtensor(x.k,x.dims,x.strides,x.asize,0){
      x.to_device(0);
      for(int i=0; i<asize; i++) arr[i]=x.arr[i];
    }
        

  public: // ---- Views --------------------------------------------------------------------------------------


    Gtensor(const view_flag& flag, Gtensor& x, const Gdims& _dims):
      k(_dims.size()), dims(_dims), strides(_dims.size()), 
      arr(x.arr), arrg(x.arrg), is_view(true), dev(x.dev){
      memsize=x.memsize; 
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--)
	strides[i]=strides[i+1]*dims[i+1];
      asize=strides[0]*dims[0];
      assert(asize==x.asize);
    }
      
    
    Gtensor<TYPE> flat_view(){
      return Gtensor<TYPE>(flag::view,*this,{asize});
    }
    
    Gtensor<TYPE> reshape_view(const Gdims& _dims){
      return Gtensor<TYPE>(flag::view,*this,_dims);
    }

    

  public: // ---- Transporters ------------------------------------------------------------------------------
 

    const Gtensor<TYPE>& to_device(const struct device& __dev) const{
      const int _dev=__dev.id();
      if(_dev==0){
 	if(dev==0) return *this;
 	delete[] arr;
	reallocate(_dev);
	//const_cast<Gtensor<TYPE>*>(this)->reallocate(_dev);
	CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arrg));
	const_cast<Gtensor<TYPE>*>(this)->arrg=nullptr;
	dev=0;
	return *this;
      }
      if(_dev>0){
	if(dev==_dev) return *this;
#ifdef _WITH_CUDA
	if(arrg) CUDA_SAFE(cudaFree(arrg));
#endif 
	reallocate(_dev);
	//const_cast<GenTensor<k,TYPE>*>(this)->reallocate(_dev);
	CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(TYPE),cudaMemcpyHostToDevice));  
	delete[] arr;
	const_cast<Gtensor<TYPE>*>(this)->arr=nullptr;
	dev=_dev;
	return *this;
      }
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size(const int i) const{
      return dims[i];
    }

    TYPE operator()(const Gindex& ix) const{
      if(dev>0) CNINE_CPUONLY();
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return arr[t];
    }

    TYPE& operator()(const Gindex& ix){
      if(dev>0) CNINE_CPUONLY();
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return arr[t];
    }

    TYPE get(const Gindex& ix) const{
      if(dev>0) CNINE_CPUONLY();
      CNINE_ASSERT(dev==0,"Gtensor::get(...) not implemented for GPU.\n");
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      return arr[t];
    }
    
    void set(const Gindex& ix, const TYPE& v){
      if(dev>0) CNINE_CPUONLY();
      CNINE_ASSERT(dev==0,"Gtensor::set(...) not implemented for GPU.\n");
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]=v;
    }

    void inc(const Gindex& ix, const TYPE& v){
      if(dev>0) CNINE_CPUONLY();
      CNINE_ASSERT(dev==0,"Gtensor::set(...) not implemented for GPU.\n");
      int t=0; for(int i=0; i<k; i++) t+=ix[i]*strides[i];
      arr[t]+=v;
    }

    void set(const TYPE& v){
      if(dev>0) CNINE_CPUONLY();
      CNINE_ASSERT(dev==0,"Gtensor::set(...) not implemented for GPU.\n");
      if(dev==0){
	std::fill(arr,arr+memsize,v);
	return;
      }
      if(dev==1){
	CNINE_UNIMPL();
	//if(typeid(TYPE)==typeid(complex<float>)) 
	//CUDA_SAFE(cudaMemset(arrg,make_cuComplex(v.real(),v.imag()),asize*sizeof(TYPE)));
	//CUDA_SAFE(cudaMemset(arrg,0,asize*sizeof(TYPE)));
	return;
      }
    }	

    TYPE get(const Gindex& ix1, const Gindex& ix2) const{
      if(dev>0) CNINE_CPUONLY();
      CNINE_ASSERT(dev==0,"Gtensor::get(...) not implemented for GPU.\n");
      const int u=ix1.size(); 
      int t=0; for(int i=0; i<u; i++) t+=ix1[i]*strides[i];
      for(int i=u; i<k; i++) t+=ix2[i-u]*strides[i];
      return arr[t];
    }
    
    void set(const Gindex& ix1, const Gindex& ix2, const TYPE& v){
      if(dev>0) CNINE_CPUONLY();
      CNINE_ASSERT(dev==0,"Gtensor::set(...) not implemented for GPU.\n");
      const int u=ix1.size(); 
      int t=0; for(int i=0; i<u; i++) t+=ix1[i]*strides[i];
      for(int i=u; i<k; i++) t+=ix2[i-u]*strides[i];
      arr[t]=v;
    }

    void inc(const Gindex& ix1, const Gindex& ix2, const TYPE& v){
      if(dev>0) CNINE_CPUONLY();
      CNINE_ASSERT(dev==0,"Gtensor::set(...) not implemented for GPU.\n");
      const int u=ix1.size(); 
      int t=0; for(int i=0; i<u; i++) t+=ix1[i]*strides[i];
      for(int i=u; i<k; i++) t+=ix2[i-u]*strides[i];
      arr[t]+=v;
    }


  public: // Arrayed k=1 special cases


    TYPE operator()(const Gindex& ix, const int i1) const{
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(ix.size()==k-1);
      int t=0; for(int i=0; i<k-1; i++) t+=ix[i]*strides[i]; 
      t+=i1*strides[k-1];
      return arr[t];
    }

    TYPE& operator()(const Gindex& ix, const int i1){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(ix.size()==k-1);
      int t=0; for(int i=0; i<k-1; i++) t+=ix[i]*strides[i];
      t+=i1*strides[k-1];
      return arr[t];
    }

    TYPE get(const Gindex& ix, const int i1) const{
      CNINE_ASSERT(dev==0,"Gtensor::get(...) not implemented for GPU.\n");
      assert(ix.size()==k-1);
      int t=0; for(int i=0; i<k-1; i++) t+=ix[i]*strides[i];
      t+=i1*strides[k-1];
      return arr[t];
    }
    
    void set(const Gindex& ix, const int i1, const TYPE& v){
      CNINE_ASSERT(dev==0,"Gtensor::set(...) not implemented for GPU.\n");
      assert(ix.size()==k-1);
      int t=0; for(int i=0; i<k-1; i++) t+=ix[i]*strides[i];
      t+=i1*strides[k-1];
      arr[t]=v;
    }


  public: // Arrayed k=2 special cases


    TYPE operator()(const Gindex& ix, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(ix.size()==k-2);
      int t=0; for(int i=0; i<k-2; i++) t+=ix[i]*strides[i]; 
      t+=i1*strides[k-2]+i2*strides[k-1];
      return arr[t];
    }

    TYPE& operator()(const Gindex& ix, const int i1, const int i2){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(ix.size()==k-2);
      int t=0; for(int i=0; i<k-2; i++) t+=ix[i]*strides[i];
      t+=i1*strides[k-2]+i2*strides[k-1];
      return arr[t];
    }

    TYPE get(const Gindex& ix, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0,"Gtensor::get(...) not implemented for GPU.\n");
      assert(ix.size()==k-2);
      int t=0; for(int i=0; i<k-2; i++) t+=ix[i]*strides[i];
      t+=i1*strides[k-2]+i2*strides[k-1];
      return arr[t];
    }
    
    void set(const Gindex& ix, const int i1, const int i2, const TYPE& v){
      CNINE_ASSERT(dev==0,"Gtensor::set(...) not implemented for GPU.\n");
      assert(ix.size()==k-2);
      int t=0; for(int i=0; i<k-2; i++) t+=ix[i]*strides[i];
      t+=i1*strides[k-2]+i2*strides[k-1];
      arr[t]=v;
    }


  public: // Arrayed k=3 special cases


    TYPE operator()(const Gindex& ix, const int i1, const int i2, const int i3) const{
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(ix.size()==k-3);
      int t=0; for(int i=0; i<k-3; i++) t+=ix[i]*strides[i]; 
      t+=i1*strides[k-3]+i2*strides[k-2]+i3*strides[k-1];
      return arr[t];
    }

    TYPE& operator()(const Gindex& ix, const int i1, const int i2, const int i3){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(ix.size()==k-3);
      int t=0; for(int i=0; i<k-3; i++) t+=ix[i]*strides[i];
      t+=i1*strides[k-3]+i2*strides[k-2]+i3*strides[k-1];
      return arr[t];
    }

    TYPE get(const Gindex& ix, const int i1, const int i2, const int i3) const{
      CNINE_ASSERT(dev==0,"Gtensor::get(...) not implemented for GPU.\n");
      assert(ix.size()==k-3);
      int t=0; for(int i=0; i<k-3; i++) t+=ix[i]*strides[i];
      t+=i1*strides[k-3]+i2*strides[k-2]+i3*strides[k-1];
      return arr[t];
    }
    
    void set(const Gindex& ix, const int i1, const int i2, const int i3, const TYPE& v){
      CNINE_ASSERT(dev==0,"Gtensor::set(...) not implemented for GPU.\n");
      assert(ix.size()==k-3);
      int t=0; for(int i=0; i<k-3; i++) t+=ix[i]*strides[i];
      t+=i1*strides[k-3]+i2*strides[k-2]+i3*strides[k-1];
      arr[t]=v;
    }


  public: // k=1 special cases


    TYPE operator()(const int i0) const{
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==1);
      return arr[i0*strides[0]];
    }

    TYPE& operator()(const int i0){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==1);
      return arr[i0*strides[0]];
    }

    TYPE& get(const int i0) const{
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==1);
      return arr[i0*strides[0]];
    }

    void set(const int i0, const TYPE x){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==1);
      arr[i0*strides[0]]=x;
    }


  public: // k=2 special cases


    TYPE operator()(const int i0, const int i1) const{
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==2);
      return arr[i0*strides[0]+i1*strides[1]];
    }

    TYPE& operator()(const int i0, const int i1){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==2);
      return arr[i0*strides[0]+i1*strides[1]];
    }

    TYPE get(const int i0, const int i1) const{
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==2);
      return arr[i0*strides[0]+i1*strides[1]];
    }

    void set(const int i0, const int i1, const TYPE x){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==2);
      arr[i0*strides[0]+i1*strides[1]]=x;
    }


  public: // k=3 special cases


    TYPE operator()(const int i0, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      assert(k==3);
      //cout<<i0*strides[0]+i1*strides[1]+i2*strides[2]<<endl; 
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]];
    }

    TYPE& operator()(const int i0, const int i1, const int i2){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==3);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]];
    }

    TYPE get(const int i0, const int i1, const int i2) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      assert(k==3);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]];
    }

    void set(const int i0, const int i1, const int i2, const TYPE x){
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      assert(k==3);
      arr[i0*strides[0]+i1*strides[1]+i2*strides[2]]=x;
    }


  public: // k=4 special cases


    TYPE operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      assert(k==4);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]];
    }

    TYPE& operator()(const int i0, const int i1, const int i2, const int i3){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==4);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]];
    }

    TYPE get(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      assert(k==4);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]];
    }

    void set(const int i0, const int i1, const int i2, const int i3, const TYPE x){
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      assert(k==4);
      arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]]=x;
    }

    void inc(const int i0, const int i1, const int i2, const int i3, const TYPE x){
      CNINE_CPUONLY(); 
      assert(k==4);
      arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]]+=x;
    }


  public: // k=5 special cases


    TYPE operator()(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      assert(k==5);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]];
    }

    TYPE& operator()(const int i0, const int i1, const int i2, const int i3, const int i4){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      assert(k==5);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]];
    }

    TYPE get(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      assert(k==5);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]];
    }

    void set(const int i0, const int i1, const int i2, const int i3, const int i4, const TYPE x){
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      assert(k==5);
      arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]]=x;
    }

    void inc(const int i0, const int i1, const int i2, const int i3, const int i4, const TYPE x){
      CNINE_CPUONLY(); 
      assert(k==5);
      arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]]+=x;
    }


  public: // k=6 special cases


    TYPE operator()(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      CNINE_ASSRT(k==6);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]];
    }

    TYPE& operator()(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      CNINE_ASSRT(k==6);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]];
    }

    TYPE get(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      CNINE_ASSRT(k==6);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]];
    }

    void set(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const TYPE x){
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      CNINE_ASSRT(k==6);
      arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]]=x;
    }

    void inc(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const TYPE x){
      CNINE_CPUONLY(); 
      CNINE_ASSRT(k==6);
      arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]]+=x;
    }


  public: // k=7 special cases


    TYPE operator()(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      CNINE_ASSRT(k==7);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]+i6*strides[6]];
    }

    TYPE& operator()(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6){
      CNINE_ASSERT(dev==0,"Gtensor::operator() not implemented for GPU.\n");
      CNINE_ASSRT(k==7);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]+i6*strides[6]];
    }

    TYPE get(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6) const{
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      CNINE_ASSRT(k==7);
      return arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]+i6*strides[6]];
    }

    void set(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const TYPE x){
      CNINE_ASSERT(dev==0, "Gtensor::operator() not implemented for GPU.\n");
      CNINE_ASSRT(k==7);
      arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]+i6*strides[6]]=x;
    }

    void inc(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const TYPE x){
      CNINE_CPUONLY(); 
      CNINE_ASSRT(k==7);
      arr[i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4]+i5*strides[5]+i6*strides[6]]+=x;
    }


  public: // ---- Slices ------------------------------------------------------------------------------------


    Gtensor<TYPE> slice(const Gdims& adims) const{
      if(dev>0) CNINE_CPUONLY();
      Gtensor<TYPE> R(dims.chunk(adims.size()),fill::raw);
      int t=0; for(int i=0; i<adims.size(); i++) t+=adims[i]*strides[i];
      std::copy(arr+t,arr+t+R.dims.asize(),R.arr);
      return R;
    }

    void set_slice(const Gindex& ix, const Gtensor<TYPE>& x) const{
      if(dev>0) CNINE_CPUONLY();
      int offs=0;
      int k0=ix.size();
      for(int i=0; i<k0; i++)
	offs+=ix[i]*strides[i];
      std::copy(x.arr,x.arr+x.asize,arr+offs);
    }


  public: // ---- Elementwise Operations --------------------------------------------------------------------


    void zero(){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    void set(const fill_gaussian& dummy){
      to_device(0);
      normal_distribution<double> distr;
      for(int i=0; i<asize; i++) arr[i]=distr(rndGen);
    }


    bool operator==(const Gtensor<TYPE>& x) const{
      CNINE_CPUONLY();
      if(x.asize!=asize) return false; 
      for(int i=0; i<asize; i++)
	if(arr[i]!=x.arr[i]) return false;
      return true;
    }

    Gtensor plus(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(asize==x.asize);
      Gtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=arr[i]+x.arr[i];
      return R;
    }

    //Gtensor<TYPE>* plusp(const Gtensor& x) const{
    //assert(asize==x.asize);
    //Gtensor* R=new Gtensor(dims,fill::raw);
    //for(int i=0; i<asize; i++) R->arr[i]=arr[i]+x.arr[i];
    //return R;
    //}

    Gtensor minus(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(asize==x.asize);
      Gtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=arr[i]-x.arr[i];
      return R;
    }

    Gtensor times(const TYPE c) const{
      CNINE_CPUONLY();
      Gtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=c*arr[i];
      return R;
    }

    Gtensor elementwise_times(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(asize==x.asize);
      Gtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=arr[i]*x.arr[i];
      return R;
    }

    Gtensor elementwise_divide(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(asize==x.asize);
      Gtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=arr[i]/x.arr[i];
      return R;
    }

    Gtensor elementwise_pow(const TYPE p, const TYPE c=1.0) const{
      CNINE_CPUONLY();
      Gtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=c*pow(arr[i],p);
      return R;
    }

    Gtensor abs() const{
      CNINE_CPUONLY();
      Gtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=std::abs(arr[i]);
      return R;
    }

    Gtensor conj() const{
      CNINE_CPUONLY();
      Gtensor R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=std::conj(arr[i]);
      return R;
    }

    Gtensor<float> real() const{
      CNINE_CPUONLY();
      Gtensor<float> R(dims,fill::raw);
      for(int i=0; i<asize; i++) R.arr[i]=std::real(arr[i]);
      return R;
    }


  public: // ---- Contractive products -----------------------------------------------------------------------


    Gtensor operator*(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(dims.size()==2);
      assert(dims[1]==x.dims[0]);

      if(x.dims.size()==1){
	Gtensor R(Gdims({dims[0]}));
	  for(int i=0; i<dims[0]; i++){
	    TYPE t=0;
	    for(int p=0; p<dims[1]; p++)
	      t+=(*this)(i,p)*x(p);
	    R(i)=t;
	  }
	return R;
      }

      assert(x.dims.size()==2);
      Gtensor R(Gdims({dims[0],x.dims[1]}));
      for(int i=0; i<dims[0]; i++)
	for(int j=0; j<x.dims[1]; j++){
	  TYPE t=0;
	  for(int p=0; p<dims[1]; p++)
	    t+=(*this)(i,p)*x(p,j);
	  R(i,j)=t;
	}
      return R;
    }


    template<typename TYPE2>
    Gtensor operator*(const Gtensor<TYPE2>& x) const{
      CNINE_CPUONLY();
      assert(dims.size()==2);
      assert(dims[1]==x.dims[0]);

      if(x.dims.size()==1){
	Gtensor R(Gdims({dims[0]}));
	  for(int i=0; i<dims[0]; i++){
	    TYPE t=0;
	    for(int p=0; p<dims[1]; p++)
	      t+=(*this)(i,p)*x(p);
	    R(i)=t;
	  }
	return R;
      }

      assert(x.dims.size()==2);
      Gtensor R(Gdims({dims[0],x.dims[1]}));
      for(int i=0; i<dims[0]; i++)
	for(int j=0; j<x.dims[1]; j++){
	  TYPE t=0;
	  for(int p=0; p<dims[1]; p++)
	    t+=(*this)(i,p)*x(p,j);
	  R(i,j)=t;
	}
      return R;
    }


    Gtensor matmul_TA(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(dims.size()==2);
      assert(x.dims.size()==2);
      assert(dims[0]==x.dims[0]);
      const int dim0=dims[1];
      const int dim1=x.dims[1];
      const int P=dims[0];
      Gtensor R(Gdims({dim0,dim1}));

      for(int i=0; i<dim0; i++)
	for(int j=0; j<dim1; j++){
	  TYPE t=0;
	  for(int p=0; p<P; p++)
	    t+=(*this)(p,i)*x(p,j);
	  R(i,j)=t;
	}
      return R;
    }

    Gtensor contract_0_0(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(dims.size()==2);
      assert(x.dims.size()==2);
      assert(dims[0]==x.dims[0]);
      const int dim0=dims[1];
      const int dim1=x.dims[1];
      const int P=dims[0];
      Gtensor R(Gdims({dim0,dim1}));

      for(int i=0; i<dim0; i++)
	for(int j=0; j<dim1; j++){
	  TYPE t=0;
	  for(int p=0; p<P; p++)
	    t+=(*this)(p,i)*x(p,j);
	  R(i,j)=t;
	}
      return R;
    }

    //#include "Gtensor_products.hpp"

  public: // ---- Operations --------------------------------------------------------------------------------


    Gtensor<float> operator-(const Gtensor<float>& x){
      CNINE_CPUONLY();
      assert(asize==x.asize);
      //to_device(0);
      //x.to_device(0);
      Gtensor<float> R(dims,fill::raw);
      for(int i=0; i<asize; i++)
	R.arr[i]=arr[i]-x.arr[i];
      return R;
    }

    TYPE norm2() const{
      CNINE_CPUONLY();
      TYPE t=0; 
      for(int i=0; i<asize; i++) 
	t+=arr[i]*arr[i]; 
      return t;
    }

    template<typename SUB>
    SUB norm2c() const{
      CNINE_CPUONLY();
      SUB t=0; 
      for(int i=0; i<asize; i++) 
	t+=std::norm(arr[i]); //*std::abs(arr[i]);
      return t;
    }

    TYPE inp(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(asize==x.asize);
      assert(dev==x.dev);
      assert(dev==0);
      TYPE t=0; 
      for(int i=0; i<asize; i++) 
	t+=arr[i]*x.arr[i];
      return t;
    }

    TYPE inpc(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(asize==x.asize);
      assert(dev==x.dev);
      assert(dev==0);
      TYPE t=0; 
      for(int i=0; i<asize; i++) 
	t+=arr[i]*std::conj(x.arr[i]);
      return t;
    }

    TYPE diff2(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(x.dims==dims);
      assert(x.asize==asize);
      TYPE t=0;
      for(int i=0; i<asize; i++)
	t+=(arr[i]-x.arr[i])*(arr[i]-x.arr[i]);
      return t;
    }

    template<typename RET>
    RET diff2c(const Gtensor& x) const{
      CNINE_CPUONLY();
      assert(x.dims==dims);
      assert(x.asize==asize);
      RET t=0;
      for(int i=0; i<asize; i++)
	t+=norm(arr[i]-x.arr[i]);
      return t;
    }

    Gtensor<TYPE> odot(const Gtensor<TYPE>& x) const{
      CNINE_ASSERT(dev==0,"GenTensor<k,TYPE>::odot(...) not implemented for GPU.\n");
      assert(dims==x.dims);
      Gtensor<TYPE> R(dims);
      for(int i=0; i<asize; i++) 
	R.arr[i]=arr[i]*x.arr[i];
      return R;
    }

    Gtensor<TYPE> odotc(const Gtensor<TYPE>& x) const{
      CNINE_ASSERT(dev==0,"GenTensor<k,TYPE>::odotc(...) not implemented for GPU.\n");
      assert(dims==x.dims);
      Gtensor<TYPE> R(dims);
      for(int i=0; i<asize; i++) 
	R.arr[i]=arr[i]*std::conj(x.arr[i]);
      return R;
    }

    Gtensor<TYPE> ReLU() const{
      CNINE_ASSERT(dev==0,"GenTensor<k,TYPE>::ReLU() not implemented for GPU.\n");
      Gtensor<TYPE> R(dims);
      CNINE_UNIMPL();
      //for(int i=0; i<asize; i++) 
      //R.arr[i]=(arr[i]>0)*arr[i];
      return R;
    }



  public: // ---- In-place operations ----------------------------------------------------------------------


    void operator+=(const Gtensor& x){
      CNINE_CPUONLY();
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
    }

    void operator-=(const Gtensor& x){
      CNINE_CPUONLY();
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]-=x.arr[i];
    }

    void operator*=(const Gtensor& x){
      CNINE_CPUONLY();
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]*=x.arr[i];
    }

    //void increment(const Gtensor& x, const TYPE c){
    //CNINE_CPUONLY();
    //assert(asize==x.asize);
    //for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
    //}

    void operator*=(const TYPE x){
      CNINE_CPUONLY();
      for(int i=0; i<asize; i++) arr[i]*=x;
    }

    void operator/=(const TYPE x){
      CNINE_CPUONLY();
      for(int i=0; i<asize; i++) arr[i]/=x;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const Gtensor& x){
      CNINE_CPUONLY();
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]+=x.arr[i];
    }

    void add(const Gtensor& x, const TYPE c){
      CNINE_CPUONLY();
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]+=x.arr[i]*c;
    }

    void addc(const Gtensor& x){
      CNINE_CPUONLY();
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]+=std::conj(x.arr[i]);
    }

    void addc(const Gtensor& x, const TYPE c){
      CNINE_CPUONLY();
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]+=std::conj(x.arr[i])*c;
    }

    void subc(const Gtensor& x, const TYPE c){
      CNINE_CPUONLY();
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]-=std::conj(x.arr[i])*c;
    }

    void subtract(const Gtensor& x){
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]-=x.arr[i];
    }

    void subtract(const Gtensor& x, const TYPE c){
      assert(asize==x.asize);
      for(int i=0; i<asize; i++) arr[i]-=x.arr[i]*c;
    }

    void add_odot(const Gtensor<TYPE>& x, const Gtensor<TYPE>& y){
      assert(x.asize==asize);
      assert(y.asize==asize);
      for(int i=0; i<asize; i++)
	arr[i]+=x.arr[i]*y.arr[i];
    }

    void add_odotc(const Gtensor<TYPE>& x, const Gtensor<TYPE>& y){
      assert(x.asize==asize);
      assert(y.asize==asize);
      for(int i=0; i<asize; i++)
	arr[i]+=x.arr[i]*std::conj(y.arr[i]);
    }

    void add_matmul(const Gtensor<TYPE>& x, const Gtensor<TYPE>& y){
      assert(k==x.k+y.k-2);
      const int I=x.asize/x.dims[x.k-1];
      const int J=y.asize/y.dims[0];
      const int K=x.dims[x.k-1];
      assert(y.dims[0]==K);
      assert(asize==I*J);

      const int istridex=x.strides[x.k-2];
      const int istride=strides[x.k-2];
      const int pstridey=y.strides[0];

      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++){
	  TYPE t=0;
	  for(int p=0; p<K; p++)
	    t+=x.arr[i*istridex+p]*y.arr[p*pstridey+j];
	  arr[istride*i+j]+=t;
	}
    }

    void add_matmul_AH(const Gtensor<TYPE>& g, const Gtensor<TYPE>& y){
      assert(g.k==k+y.k-2);
      const int I=asize/dims[k-1];
      const int J=y.asize/y.dims[0];
      const int K=dims[k-1];
      assert(y.dims[0]==K);
      assert(g.asize==I*J);

      const int istridex=strides[k-2];
      const int istride=g.strides[k-2];
      const int pstridey=y.strides[0];

      for(int i=0; i<I; i++)
	for(int p=0; p<K; p++){
	  TYPE t=0;
	  for(int j=0; j<J; j++)
	    t+=g.arr[istride*i+j]*std::conj(y.arr[p*pstridey+j]);
	  arr[istridex*i+p]+=t;
	}
    }

    void add_matmul_HA(const Gtensor<TYPE>& x, const Gtensor<TYPE>& g){
      assert(g.k==x.k+k-2);
      const int I=x.asize/x.dims[x.k-1];
      const int J=asize/dims[0];
      const int K=x.dims[x.k-1];
      assert(dims[0]==K);
      assert(g.asize==I*J);

      const int istridex=x.strides[x.k-2];
      const int istride=g.strides[x.k-2];
      const int pstridey=strides[0];

      for(int p=0; p<K; p++)
	for(int j=0; j<J; j++){
	  TYPE t=0;
	  for(int i=0; i<I; i++)
	    t+=std::conj(x.arr[i*istridex+p])*g.arr[istride*i+j];
	  arr[p*pstridey+j]+=t;
	}
    }


    // Obsolete 
    /*
    void incprod(const Gtensor<TYPE>& x, const Gtensor<TYPE>& y){
      assert(dims.size()==2);
      assert(x.dims.size()==2);
      assert(y.dims.size()==2);

      const int I=dims[0];
      assert(x.dims[0]==I);
      const int J=dims[1];
      assert(y.dims[1]==J);
      const int K=x.dims[1];
      assert(y.dims[0]==K);

      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++){
	  TYPE t=0;
	  for(int p=0; p<K; p++)
	    t+=x(i,p)*y(p,j);
	  (*this)(i,j)+=t;
	}
    }
    */


  public: // ---- Optimization kernels ------------------------------------------------------------------------

    /*
    void adam_update(Gtensor<TYPE>& g, Gtensor<TYPE>& mt, Gtensor<TYPE>& vt, 
      const TYPE beta1, const TYPE beta2, const TYPE alpha, const TYPE epsilon){
      assert(g.asize==asize);
      assert(mt.asize==asize);
      assert(vt.asize==asize);

      if(dev==0){
	assert(g.dev==0);
	assert(mt.dev==0);
	assert(vt.dev==0);
	for(int i=0; i<asize; i++){  
	  mt.arr[i]=beta1*mt.arr[i]+g.arr[i]*(static_cast<TYPE>(1.0)-beta1);
	  vt.arr[i]=beta2*vt.arr[i]+std::pow(g.arr[i],2)*(static_cast<TYPE>(1.0)-beta2);  
	  arr[i]-=alpha*mt.arr[i]/(sqrt(vt.arr[i])+epsilon);
	}
      }else{
	CNINE_UNIMPL();
      }
      
    }
    */

    /*
    void adam_update_complex(Gtensor<TYPE>& g, Gtensor<TYPE>& mt, Gtensor<TYPE>& vt, 
      const float beta1, const float beta2, const float alpha, const float epsilon){
      assert(g.asize==asize);
      assert(mt.asize==asize);
      assert(vt.asize==asize);

      if(dev==0){
	assert(g.dev==0);
	assert(mt.dev==0);
	assert(vt.dev==0);
	for(int i=0; i<asize; i++){ 
	  mt.arr[i]=beta1*mt.arr[i]+g.arr[i]*(static_cast<TYPE>(1.0)-beta1);
	  vt.arr[i]=beta2*vt.arr[i]+complex<float>(pow(g.arr[i].real(),2),pow(g.arr[i].imag(),2))*(static_cast<TYPE>(1.0)-beta2);
	  arr[i]-=complex<float>(alpha*mt.arr[i].real()/(sqrt(vt.arr[i].real())+epsilon),
	    alpha*mt.arr[i].imag()/(sqrt(vt.arr[i].imag())+epsilon));
	}
      }else{
	FCG_UNIMPL();
      }
      
    }
    */

    /*
    template<typename BTYPE>
    void adagrad_update_complex(const Gtensor<complex<BTYPE> >& g, Gtensor<complex<BTYPE> >& G, 
      const BTYPE eta, const BTYPE epsilon){
      assert(g.asize==asize);
      assert(G.asize==asize);

      if(dev==0){
	assert(g.dev==0);
	assert(G.dev==0);

	for(int i=0; i<asize; i++){ 
	  const BTYPE gr=g.arr[i].real();
	  const BTYPE gi=g.arr[i].imag();
	  
	  G.arr[i]+=complex<BTYPE>(gr*gr,gi*gi);
	  arr[i]-=eta*complex<BTYPE>(pow(epsilon+G.arr[i].real(),-0.5)*gr,pow(epsilon+G.arr[i].imag(),-0.5)*gi);
	}
      }else{
	FCG_UNIMPL();
      }

    }

    */



  public: // ---- I/O ----------------------------------------------------------------------------------------

    /*
    Gtensor(const string filename, const device& dev=0){
      ifstream ifs(filename.c_str());
      ifs.read(reinterpret_cast<char*>(&k),sizeof(int));
      dims.resize(k);
      for(int i=0; i<k; i++)
	ifs.read(reinterpret_cast<char*>(&dims[i]),sizeof(int));
      make_strides();
      arr=new TYPE[asize];
      ifs.read(reinterpret_cast<char*>(arr),asize*sizeof(TYPE));
      to_device(dev);
      ifs.close();
    }
    */

    /*
    int save(const string filename) const{
      ofstream ofs(filename.c_str());
      ofs.write(reinterpret_cast<const char*>(&k),sizeof(int));
      for(int i=0; i<k; i++)
	ofs.write(reinterpret_cast<const char*>(&dims[i]),sizeof(int));
      if(dev==0)
	ofs.write(reinterpret_cast<const char*>(arr),asize*sizeof(TYPE));
      else{
	Gtensor<TYPE> T(*this,device(0));
	ofs.write(reinterpret_cast<const char*>(T.arr),asize*sizeof(TYPE));
      }
      ofs.close();
      return 0;
    }
    */

    /*
    Gtensor(const string filename, const device& dev=0){
      Bifstream ifs(filename);
      Gtensor T(ifs); 
      (*this)=std::move(T);//Gtensor(ifs);
    }    

    void save(const string filename) const{
      Bofstream ofs(filename);
      serialize(ofs);
    }

    Gtensor(Bifstream& ifs){
      ifs.read(ak);
      Gdims fdims(ifs); 
      dims=fdims; 
      make_strides(ak);
      reallocate();
      ifs.read_array(arr);
    }

    void serialize(Bofstream& ofs) const{
      ofs.write(ak); 
      dims.serialize(ofs);
      ofs.write_array(arr,asize);
    }
    */

    string str(const string indent="", const float eps=0) const{
      //if(dev>0) return Gtensor(*this,typename device(0)).str(indent,eps);
      assert(dev==0);
      ostringstream oss;

      if(k==1){
	oss<<base_indent<<indent<<"[ ";
	for(int j=0; j<dims[0]; j++)
	  if(eps==0) oss<<arr[j]<<" ";
	  else 
	    if(std::abs(arr[j])>eps) oss<<arr[j]<<" ";
	      else oss<<TYPE()<<" ";
	oss<<"]";
	oss<<"\n";
      }

      if(k==2){
	for(int i=0; i<dims[0]; i++){
	  oss<<base_indent<<indent<<"[ ";
	  for(int j=0; j<dims[1]; j++)
	    //oss<<(*this)({i,j})<<" ";
	    if(eps==0) oss<<(*this)({i,j})<<" ";
	    else 
	      if(std::abs((*this)({i,j}))>eps) oss<<(*this)({i,j})<<" ";
		else oss<<TYPE()<<" ";
		oss<<"]";
	  if(i<dims[0]-1) oss<<"\n";
	}
	oss<<"\n";
      }

      if(k==3 && dims[2]==1){
	for(int i=0; i<dims[0]; i++){
	  oss<<indent<<"[ ";
	  for(int j=0; j<dims[1]; j++)
	    oss<<(*this)({i,j,0})<<" ";
	  oss<<"]";
	  if(i<dims[0]-1) oss<<"\n";
	}
	oss<<"\n";
	return oss.str(); 
      }

      if(k==3){
	for(int u=0; u<dims[0]; u++){
	  for(int i=0; i<dims[1]; i++){
	    oss<<indent<<"[ ";
	    for(int j=0; j<dims[2]; j++)
	      oss<<(*this)({u,i,j})<<" ";
	    oss<<"]";
	    oss<<"\n";
	  }
	  oss<<"\n";
	}
	return oss.str();  
      }

      if(k==4){
	for(int u=0; u<dims[0]; u++){
	  for(int v=0; v<dims[1]; v++){
	    for(int i=0; i<dims[2]; i++){
	      oss<<indent<<"[ ";
	      for(int j=0; j<dims[3]; j++)
		oss<<(*this)({u,v,i,j})<<" ";
	      oss<<"]";
	      oss<<"\n";
	    }
	    oss<<"\n";
	  }
	  oss<<"\n";
	}
	return oss.str();  
      }

      if(k==5){
	for(int i0=0; i0<dims[0]; i0++){
	  for(int i1=0; i1<dims[1]; i1++){
	    for(int i2=0; i2<dims[2]; i2++){
	      for(int i3=0; i3<dims[3]; i3++){
	      oss<<indent<<"[ ";
	      for(int j=0; j<dims[4]; j++)
		oss<<(*this)({i0,i1,i2,i3,j})<<" ";
	      oss<<"]";
	      oss<<"\n";
	      }
	      oss<<"\n";
	    }
	    oss<<"\n";
	  }
	  oss<<"\n";
	}
	return oss.str();  
      }

      if(k==6){
	for(int i0=0; i0<dims[0]; i0++){
	  for(int i1=0; i1<dims[1]; i1++){
	    for(int i2=0; i2<dims[2]; i2++){
	      for(int i3=0; i3<dims[3]; i3++){
		for(int i4=0; i4<dims[4]; i4++){
		  oss<<indent<<"[ ";
		  for(int j=0; j<dims[5]; j++)
		    oss<<(*this)({i0,i1,i2,i3,i4,j})<<" ";
		  oss<<"]";
		  oss<<"\n";
		}
		oss<<"\n";
	      }
	      oss<<"\n";
	    }
	    oss<<"\n";
	  }
	  oss<<"\n";
	}
	return oss.str();  
      }

      if(k==7){
	for(int i0=0; i0<dims[0]; i0++){
	  for(int i1=0; i1<dims[1]; i1++){
	    for(int i2=0; i2<dims[2]; i2++){
	      for(int i3=0; i3<dims[3]; i3++){
		for(int i4=0; i4<dims[4]; i4++){
		  for(int i5=0; i4<dims[5]; i5++){
		    oss<<indent<<"[ ";
		    for(int j=0; j<dims[6]; j++)
		      oss<<(*this)({i0,i1,i2,i3,i4,i5,j})<<" ";
		    oss<<"]";
		    oss<<"\n";
		  }
		  oss<<"\n";
		}
		oss<<"\n";
	      }
	      oss<<"\n";
	    }
	    oss<<"\n";
	  }
	  oss<<"\n";
	}
	return oss.str();  
      }

      /*
      if(k>2){
	for(int offs=0; offs<asize; offs+=strides[k-3]){
	  Index<k-2> ix(coords(offs),0);
	  oss<<indent<<"tensor("; for(int i=0; i<k-2; i++) oss<<ix[i]<<","; oss<<"*,*)=\n";
	  for(int i=0; i<dims[k-2]; i++){
	    oss<<indent<<"  [ ";
	    for(int j=0; j<dims[k-1]; j++)
	      if(abs(arr[offs+i*dims[k-1]+j])>10e-5)
		oss<<arr[offs+i*dims[k-1]+j]<<" ";
	      else oss<<"0 ";
	    oss<<"]\n";
	  }
	}
      }
      */
      
      return oss.str();
    }

    string str_transp(const string indent="") const{
      //if(dev>0) return Gtensor(*this,struct device(0)).str(indent);
      assert(dev==0);
      ostringstream oss;

      if(k==1){
	oss<<indent<<"[ ";
	for(int j=0; j<dims[0]; j++)
	  oss<<arr[j]<<" ";
	oss<<"]";
      }

      if(k==2){
	for(int i=0; i<dims[1]; i++){
	  oss<<indent<<"[ ";
	  for(int j=0; j<dims[0]; j++)
	    oss<<(*this)({j,i})<<" ";
	  oss<<"]";
	  if(i<dims[1]-1) oss<<"\n";
	}
      }

      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Gtensor<TYPE>& x){
      stream<<x.str(); return stream;
    }

  };

  
  
}

#endif 
    /*
    template<unsigned long K>
    Gtensor(const GenTensor<K,TYPE>& x): 
      Gtensor(K,Gdims(x.dims),vector<int>(K),x.asize,x.dev){
      for(int i=0; i<k; i++) strides[i]=x.strides[i];
      //COPY_WARNING;
      if(dev==0) std::copy(x.arr,x.arr+asize,arr);
      if(dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
    }
    */
      
    /*
    template<unsigned long K>
    Gtensor& operator=(const GenTensor<K,TYPE>& x){
      ASSIGN_WARNING;
      if(!is_view) delete[] arr;
      if(!is_view && arrg) cudaFree(arrg); 
      k=K; dims=x.dims; strides=x.strides; asize=x.asize; device=x.device;
      if(device==0){
	arr=new TYPE[asize]; 
	std::copy(x.arr,x.arr+asize,arr);
      }
      if(device==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, asize*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,asize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
      return *this;
    }
    */
    //Gtensor(const Gdims& _dims, const TYPE x, const device& dev=0):
    //Gtensor(_dims){
    //std::fill(arr,arr+asize,x);
      //for(int i=0; i<asize; i++) arr[i]=x;
      //to_device(dev);
    //}


