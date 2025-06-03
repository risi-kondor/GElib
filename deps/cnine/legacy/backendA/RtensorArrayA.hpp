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


#ifndef _CnineRtensorArrayA
#define _CnineRtensorArrayA

#include "Cnine_base.hpp"
#include "CnineObject.hpp"
#include "Gdims.hpp"
#include "Gtensor.hpp"
#include "RscalarA.hpp"
#include "RtensorA.hpp"

#include "Reducer.hpp"
#include "Broadcaster.hpp"
#include "RtensorA_accessor.hpp"

#include "Cmaps.hpp"
//#include "GenericOp.hpp"
#include "GenericCop.hpp"

#include "RtensorA_add_cop.hpp"
#include "RtensorA_add_plus_cop.hpp"
#include "RtensorA_add_times_c_cop.hpp"
#include "RtensorA_add_Mprod_cop.hpp"


#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  class RtensorArrayA: public RtensorA{
  public:

    int ak;
    int k;
    int nbu=-1; 

    Gdims adims;
    Gdims cdims;

    vector<int> astrides;
    vector<int> cstrides;

    int aasize;
    int asize=0;

    int cellstride;


  public:

    //RtensorArrayA(){}

    ~RtensorArrayA(){
    }
    
    string classname() const{
      return "RtensorArrayA";
    }

    string describe() const{
      return "RtensorArrayA"+adims.str()+dims.str();
    }


  private: // ---- Constructors -----------------------------------------------------------------------------


    using RtensorA::RtensorA;
    

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _dev=0): 
      RtensorA(_adims,_cdims,_dev), 
      ak(_adims.size()), 
      k(_cdims.size()), 
      adims(_adims), 
      cdims(_cdims),
      astrides(_adims.size()),
      cstrides(_cdims.size()){

      asize=strides[ak]*cdims[0];
      cellstride=roundup(asize,32);
      aasize=RtensorA::asize/cellstride;
      for(int i=0; i<ak; i++)
	astrides[i]=strides[i]/cellstride;
      for(int i=0; i<k; i++)
	cstrides[i]=strides[ak+i];
    }

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const fill_noalloc& dummy, const int _dev=0): 
      RtensorA(_adims,_cdims,dummy,_dev), 
      ak(_adims.size()), 
      k(_cdims.size()), 
      adims(_adims), 
      cdims(_cdims),
      astrides(_adims.size()),
      cstrides(_cdims.size()){

      asize=strides[ak]*cdims[0];
      cellstride=roundup(asize,32);
      aasize=RtensorA::asize/cellstride;
      for(int i=0; i<ak; i++)
	astrides[i]=strides[i]/cellstride;
      for(int i=0; i<k; i++)
	cstrides[i]=strides[ak+i];
    }


  public: // ---- Filled constructors -----------------------------------------------------------------------


    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const int _dev):
      RtensorArrayA(adims,_cdims.prepend(_nbu),_dev){
    }

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const fill_raw& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims,_dev){}
    
    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const fill_raw& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims.prepend(_nbu),_dev){}
  

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const fill_noalloc& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims.prepend(_nbu),_dev){}

  
    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const fill_zero& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims,_dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }
  
    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const fill_zero& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims.prepend(_nbu),_dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
    }

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const fill_identity& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims,_dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
      CNINE_UNIMPL()
    }
  
    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const fill_identity& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims.prepend(_nbu),_dev){
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1) CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)));
      CNINE_UNIMPL()
    }

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const fill_ones& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims,fill::raw,0){
      std::fill(arr,arr+cst,1);
      if(_dev>0) move_to_device(_dev);
    }

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const fill_ones& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims.prepend(_nbu),fill::raw,0){
      std::fill(arr,arr+cst,1);
      if(_dev>0) move_to_device(_dev);
    }

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const fill_sequential& dummy, const int _dev=0):
      RtensorArrayA(_adims,_cdims,fill::zero,0){
      for(int j=0; j<aasize; j++){
	for(int i=0; i<asize; i++) arr[i+j*cellstride]=i+j*asize;
      }
      if(_dev>0) move_to_device(_dev);
    }

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const fill_sequential& dummy, const int _dev=0):
      RtensorArrayA(_adims,_cdims.prepend(_nbu),fill::zero,0){
      for(int j=0; j<aasize; j++){
	for(int i=0; i<asize; i++) arr[i+j*cellstride]=i+j*asize;
      }
      if(dev>0) move_to_device(_dev);
    }

    /*
    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const fill_identity& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims,-1,RtensorArrayA_cop::setIdentity(),_dev){
    }

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const fill_identity& dummy, const int _dev=0): 
      RtensorArrayA(_adims,_cdims,_nbu,RtensorArrayA_cop::setIdentity(),_dev){
    }
    */

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const fill_gaussian& dummy, const int _dev=0):
      RtensorArrayA(_adims,_cdims,fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<cst; i++) arr[i]=distr(rndGen);
      move_to_device(_dev);
    }

    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const fill_gaussian& dummy, const int _dev=0):
      RtensorArrayA(_adims,_cdims.prepend(_nbu),fill::raw,0){
      normal_distribution<double> distr;
      for(int i=0; i<cst; i++) arr[i]=distr(rndGen);
      move_to_device(_dev);
    }


    RtensorArrayA(const Gdims& _adims, const RtensorA& x, const int _dev=0):
      RtensorArrayA(_adims,x.dims,fill::raw,0){
      assert(x.dev==0);
      for(int i=0; i<aasize; i++)
	set_cell(i,x);
      move_to_device(_dev);
    }


    RtensorArrayA(const RtensorA& x, const int _dev=-1):
      RtensorArrayA(Gdims(1),x.dims,fill::raw,x.get_dev()){
      set_cell(0,x);
      if(_dev>=0) move_to_device(_dev);
    }


  public: // ---- Lambda constructors ------------------------------------------------------------------------

    /*
    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const RtensorA_inplace_cop& cop, const int _dev):
      RtensorArrayA(_adims,_cdims.prepend(_nbu),0){
      for(int i=0; i<aasize; i++)
	cop(cellspec(),arr+i*cellstride,arrc+i*cellstride);
      if(dev>0) to_device(dev);
    }
    */


   RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, 
     std::function<float(const Gindex& aix, const Gindex& ix)> fn, const int dev=0):
     RtensorArrayA(_adims,_cdims.prepend(_nbu),fill::raw){
      if(nbu==-1){
	for(int i=0; i<aasize; i++){
	  Gindex aix=Gindex(i,_adims);
	  for(int j=0; j<asize; j++){
	    Gindex ix=Gindex(j,_cdims);
	    float t=fn(aix,ix);
	    arr[i*cellstride+j]=t;
	  }
	}
      }else{ // TODO 
	for(int i=0; i<aasize; i++){
	  Gindex aix=Gindex(i,_adims);
	  for(int j=0; j<asize; j++){
	    Gindex ix=Gindex(j,_cdims);
	    float t=fn(aix,ix);
	    arr[i*cellstride+j]=t;
	  }
	}
      }
      if(dev>0) move_to_device(dev);
    }

   RtensorArrayA(const Gdims& _adims, const Gdims& _cdims,
      std::function<float(const Gindex& aix, const Gindex& ix)> fn, const int dev=0):
     RtensorArrayA(_adims,_cdims,-1,fn,dev){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    RtensorArrayA(const RtensorArrayA& x):
      RtensorA(x), ak(x.ak), nbu(x.nbu), adims(x.adims), cdims(x.cdims), astrides(x.astrides), cstrides(x.cstrides), 
      aasize(x.aasize), asize(x.asize), cellstride(x.cellstride){
      CNINE_COPY_WARNING();
    }

    RtensorArrayA(RtensorArrayA&& x):
      RtensorA(std::move(x)), ak(x.ak), nbu(x.nbu), adims(x.adims), cdims(x.cdims), 
      astrides(x.astrides), cstrides(x.cstrides), 
      aasize(x.aasize), asize(x.asize), cellstride(x.cellstride){
      CNINE_MOVE_WARNING();
    }

    RtensorArrayA& operator=(const RtensorArrayA& x){
      CNINE_ASSIGN_WARNING();
      RtensorA::operator=(x);
      ak=x.ak;
      nbu=x.nbu;
      adims=x.adims;
      cdims=x.cdims;
      astrides=x.astrides;
      cstrides=x.cstrides;
      aasize=x.aasize;
      asize=x.asize;
      cellstride=x.cellstride;
      return *this;
    }

    RtensorArrayA& operator=(RtensorArrayA&& x){
      CNINE_MOVEASSIGN_WARNING();
      RtensorA::operator=(std::move(x));
      ak=x.ak;
      nbu=x.nbu;
      adims=x.adims;
      cdims=x.cdims;
      astrides=x.astrides;
      cstrides=x.cstrides;
      aasize=x.aasize;
      asize=x.asize;
      cellstride=x.cellstride;
      return *this;
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    RtensorArrayA(const RtensorArrayA& x, const int _dev):
      RtensorA(x,_dev){
      ak=x.ak;
      adims=x.adims;
      cdims=x.cdims;
      astrides=x.astrides;
      cstrides=x.cstrides;
      aasize=x.aasize;
      asize=x.asize;
      cellstride=x.cellstride;
    }

    RtensorArrayA& move_to(const device& _dev){
      RtensorA::move_to_device(_dev.id());
      return *this;
    }
    
    RtensorA& move_to_device(const int _dev){
      RtensorA::move_to_device(_dev);
      return *this;
    }
    
    RtensorArrayA to(const device& _dev) const{
      return RtensorArrayA(*this,_dev.id());
    }

    RtensorA to_device(const int _dev) const{
      return RtensorArrayA(*this,_dev);
    } 


  public: // ---- ATEN Conversions ---------------------------------------------------------------------------


#ifdef _WITH_ATEN

    static bool is_viewable(const at::Tensor& T, const int _ak){
      if(T.dim()>=_ak && T.stride(_ak)%32==0) return true;
      else return false;
    }

    RtensorArrayA(const int _ak, const at::Tensor& T){
      CNINE_CONVERT_FROM_ATEN_WARNING();

      int _n=T.dim();
      Gdims _adims(_ak,fill_raw());
      for(int i=0; i<_ak; i++) _adims[i]=T.size(i);
      Gdims _cdims(_n-_ak,fill_raw());
      for(int i=0; i<_n-_ak; i++) _cdims[i]=T.size(i+_ak);

      dev=T.type().is_cuda();
      (*this)=RtensorArrayA(_adims,_cdims,dev);

      if(T.stride(_ak-1)%32!=0){
	int tstride=T.stride(_ak-1);
	if(dev==0){
	  for(int i=0; i<aasize; i++)
	    std::copy(T.data<float>()+i*tstride,T.data<float>()+i*tstride+asize,arr+i*cellstride);
	}
	if(dev==0){
	  for(int i=0; i<aasize; i++)
	    CUDA_SAFE(cudaMemcpy(arrg+i*cellstride,T.data<float>()+i*tstride,asize*sizeof(float),cudaMemcpyDeviceToDevice));
	}
      }else{
	if(dev==0) std::copy(T.data<float>(),T.data<float>()+memsize,arr);
	if(dev==1) CUDA_SAFE(cudaMemcpy(arrg,T.data<float>(),memsize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
    }

    at::Tensor torch() const{
      CNINE_CONVERT_TO_ATEN_WARNING();

      assert(dev==0);
      vector<int64_t> v(1); 
      v[0]=cst;
      at::Tensor R(at::zeros(v,torch::CPU(at::kFloat))); 
      std::copy(arr,arr+cst,R.data<float>());

      //std::vector<long long> v(k+ak);
      //for(int i=0; i<ak; i++) v[i]=adims[i];
      //for(int i=0; i<k; i++) v[i+ak]=cdims[i];

      //std::vector<long long> v(k+ak);
      //for(int i=0; i<ak; i++) v[i]=adims[i];
      //for(int i=0; i<k; i++) v[i+ak]=cdims[i];

      vector<int64_t> _strides(RtensorA::strides.size());
      for(int i=0; i<RtensorA::strides.size(); i++)
	_strides[i]=RtensorA::strides[i];

      at::Tensor Rd=R.as_strided(RtensorA::dims.to_vec<int64_t>(),_strides);
      return Rd;
    }

#endif 


  public: // ---- Variants -----------------------------------------------------------------------------------


    RtensorArrayA(const RtensorArrayA& x, const view_flag& flag):
      RtensorA(x,flag), ak(x.ak), nbu(x.nbu), adims(x.adims), cdims(x.cdims), astrides(x.astrides), cstrides(x.cstrides), 
      aasize(x.aasize), asize(x.asize), cellstride(x.cellstride){
    }


    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const int _nbu, const fill_view& dummy, 
      float* _arr, const int _dev=0): 
      RtensorArrayA(_adims,_cdims.prepend(_nbu),fill::noalloc,_dev){
      if(dev==0){
	arr=_arr; 
      }
      if(dev==1){
	arrg=_arr; 
      }
      is_view=true;
    }
  
    RtensorArrayA(const Gdims& _adims, const Gdims& _cdims, const fill_view& dummy, 
      float* _arr, const int _dev=0): 
      RtensorArrayA(_adims,_cdims,-1,dummy,_arr,_dev){}


    template<typename FILLTYPE, typename = 
	     typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    RtensorArrayA(const RtensorArrayA& x, const Gdims& _adims, const FILLTYPE& fill):
      RtensorArrayA(_adims,x.cdims,x.get_nbu(),fill,x.get_dev()){}


    RtensorArrayA(const RtensorArrayA& x, const Gdims& _adims, const Gdims& _cdims):
      RtensorArrayA(_adims,_cdims,x.get_nbu(),fill::raw,get_dev()){
      assert(cst==x.cst);
      if(dev==0){
	std::copy(x.arr,x.arr+cst,arr);
      }
#ifdef _WITH_CUDA
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,cst*sizeof(float),cudaMemcpyDeviceToDevice));
      }
#endif
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nadims() const{
      return adims.size();
    }

    const Gdims& get_adims() const{
      return adims;
    }

    int get_adim(const int i) const{
      return adims[i];
    }

    int get_ncdims() const{
      return cdims.size();
    }

    const Gdims& get_cdims() const{
      return cdims;
    }

    int get_cdim(const int i) const{
      return cdims[i];
    }

    int cell_combined_size(const int a, const int b) const{
      assert(b<=k);
      assert(a<=b);
      if(b>0 && cstrides[b-1]==0) return 0;
      if(a>0) return (cstrides[a-1])/(cstrides[b-1]);
      if(b>0) return asize/cstrides[b-1];
      return 1; 
    }


  public: // ---- Cell Access --------------------------------------------------------------------------------


    RtensorA get_cell(const Gindex& aix) const{
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      RtensorA R(cdims,nbu,fill::raw,dev);
      copy_cell_into(R,aix);
      return R;
    }

    void copy_cell_into(RtensorA& R, const Gindex& aix) const{
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      assert(dev==R.dev);
      int t=aix(strides);
      if(dev==0){
	std::copy(arr+t,arr+t+asize,R.arr);
	return; 
      }
      CUDA_SAFE(cudaMemcpy(R.arrg,arrg+t,asize*sizeof(float),cudaMemcpyDeviceToDevice));
    }

    void add_cell_into(RtensorA& R, const Gindex& aix) const{
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      assert(dev==R.dev);
      int t=aix(strides);
      if(dev==0){
	float* p=arr+t;
	for(int i=0; i<asize; i++) R.arr[i]+=p[i];
	return; 
      }
      if(dev==1){
	const float alpha=1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, arrg+t, 1, R.arrg, 1));
      }
    }

    void set_cell(const Gindex& aix, const RtensorA& x) const{
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      int t=aix(strides);
      if(dev==0){
	std::copy(x.arr,x.arr+asize,arr+t);
	return; 
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg+t,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
    }

    void add_to_cell(const Gindex& aix, const RtensorA& x) const{
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      int t=aix(strides);
      if(dev==0){
	stdadd(x.arr,x.arr+asize,arr+t);
	return; 
      }
      if(dev==1){
	const float alpha=1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg+t, 1));
      }
    }

    void set_cell(const int ix, const RtensorA& x) const{
      int t=ix*cellstride;
      if(dev==0){
	std::copy(x.arr,x.arr+asize,arr+t);
	return; 
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg+t,x.arrg,asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
    }

    void add_to_cell(const int ix, const RtensorA& x) const{
      int t=ix*cellstride;
      if(dev==0){
	stdadd(x.arr,x.arr+asize,arr+t);
	return; 
      }
      if(dev==1){
	const float alpha=1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg+t, 1));
      }
    }



    RtensorA cell(const int i){
      //cout<<"RtensorA view"<<endl;
      return RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }

    const RtensorA cell(const int i) const{
      //cout<<"const RtensorA view"<<endl;
      return RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }

    RtensorA cell_view(const int i){
      //cout<<"RtensorA view"<<endl;
      return RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }

    RtensorA cell(const int i, const int j){
      //cout<<"RtensorA view"<<endl;
      return RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+(i*astrides[0]+j*astrides[1])*cellstride,flag::view);
    }

    const RtensorA cell(const int i, const int j) const{
      //cout<<"const RtensorA view"<<endl;
      return RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+(i*astrides[0]+j*astrides[1])*cellstride,flag::view);
    }

    RtensorA cell_view(const int i, const int j){
      //cout<<"RtensorA view"<<endl;
      return RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+(i*astrides[0]+j*astrides[1])*cellstride,flag::view);
    }

    RtensorA cell(const Gindex& aix){
      //cout<<"RtensorA view"<<endl;
      int i=aix(astrides);
      return RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }

    const RtensorA cell(const Gindex& aix) const{
      //cout<<"const RtensorA view"<<endl;
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      int i=aix(astrides);
      return RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }

    RtensorA cell_view(const Gindex& aix){
      //cout<<"RtensorA view"<<endl;
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      int i=aix(astrides);
      return RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }


    RtensorA* cellp(const int i){
      return new RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }

    const RtensorA* cellp(const int i) const{
      return new RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }

    RtensorA* cellp(const Gindex& aix){
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      int i=aix(astrides);
      return new RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }

    const RtensorA* cellp(const Gindex& aix) const{
      int i=aix(astrides);
      return new RtensorA(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,flag::view);
    }


    RtensorA_accessor accessor(const int i){
      return RtensorA_accessor(arr+i*cellstride,cstrides);
    }

    RtensorA_accessor accessor(const int i) const{
      return RtensorA_accessor(arr+i*cellstride,cstrides);
    }

    //RtensorA_spec cellspec() const{
    //return RtensorA_spec(cdims,nbu,cstrides,asize);
    //}


  public: // ---- Reshaping ----------------------------------------------------------------------------------


    RtensorArrayA arrayshape(const Gdims& _adims) const{
      return RtensorArrayA(*this,_adims,cdims);
    }

    RtensorArrayA& change_arrayshape(const Gdims& _adims){
      assert(_adims.asize()==aasize);
      int oak=ak;
      ak=_adims.size();
      adims=_adims;
      astrides.resize(ak);
      int t=1;
      for(int i=ak-1; i>=0; i--){
	astrides[i]=t;
	t*=adims[i];
      }
      dims=Gdims(adims,cdims);
      vector<int> _strides(ak+k);
      for(int i=0; i<ak; i++)
	_strides[i]=astrides[i]*cellstride;
      for(int i=0; i<ak; i++)
	_strides[ak+i]=strides[oak+i];
      strides=_strides;
      return *this; 
    }

    RtensorArrayA as_arrayshape(const Gdims& _adims) const{
      RtensorArrayA R(*this,view_flag());
      return R.change_arrayshape(_adims);
    }


  public: // ----- Cell Operations ---------------------------------------------------------------------------

    /*
    void map_inplace(const RtensorA_inplace_cop& cop){
      RtensorA_spec cspec=cellspec();
      for(int i=0; i<aasize; i++)
	cop(cspec,arr+i*cellstride,arrc+i*cellstride);
    }

    void map_unary(const RtensorA_unary_cop& cop, const RtensorArrayA& x0){
      RtensorA_spec cspec=cellspec();
      RtensorA_spec cspec0=x0.cellspec();
      assert(aasize=x0.aasize);
      for(int i=0; i<aasize; i++)
	cop(cspec,cspec0,arr+i*cellstride,arrc+i*cellstride,x0.arr+i*x0.cellstride,x0.arrc+i*x0.cellstride);
    }

    void map_binary(const RtensorA_binary_cop& cop, const RtensorArrayA& x0, const RtensorArrayA& x1){
      RtensorA_spec cspec=cellspec();
      RtensorA_spec cspec0=x0.cellspec();
      RtensorA_spec cspec1=x1.cellspec();
      assert(aasize=x0.aasize);
      assert(aasize=x1.aasize);
      for(int i=0; i<aasize; i++)
	cop(cspec,cspec0,cspec1,arr+i*cellstride,arrc+i*cellstride,
	  x0.arr+i*x0.cellstride,x0.arrc+i*x0.cellstride,x1.arr+i*x1.cellstride,x1.arrc+i*x1.cellstride);
    }

    void add_outer(const RtensorA_binary_cop& cop, const RtensorArrayA& x0, const RtensorArrayA& x1){
      const int I=x0.aasize;
      const int J=x1.aasize;
      RtensorA_spec cspec=cellspec();
      RtensorA_spec cspec0=x0.cellspec();
      RtensorA_spec cspec1=x1.cellspec();
      assert(aasize=I*J);
      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++)
	  cop(cspec,cspec0,cspec1,arr+(i*J+j)*cellstride,arrc+(i*J+j)*cellstride,
	    x0.arr+i*x0.cellstride,x0.arrc+i*x0.cellstride,x1.arr+j*x1.cellstride,x1.arrc+j*x1.cellstride);
    }
    */

  public: // ---- Map operations -----------------------------------------------------------------------------


    /*
    void map(const InplaceOp<RtensorA>& op){
      for(int i=0; i<aasize; i++){
	RtensorA t=cell(i);
	op(t);
      }
    }
    */

    /*
    void map(const UnaryCop<RtensorA,RtensorArrayA>& op, const RtensorArrayA& x0){
      for(int i=0; i<aasize; i++){
	RtensorA t=cell(i);
	op(t,x0.cell(i));
      }
    }
    */

    /*
    template<typename ARG0>
    void map(const UnaryOp<RtensorA,RtensorA>& op, const ARG0& x0){
      for(int i=0; i<aasize; i++){
	RtensorA t=cell(i);
	op(t,x0);
      }
    }

    void map(const BinaryOp<RtensorA,RtensorA,RtensorA>& op, const RtensorArrayA& x0, const RtensorArrayA& x1){
      if(dev==0){
	for(int i=0; i<aasize; i++){
	  RtensorA t=cell(i);
	  op(t,x0.cell(i),x1.cell(i));
	}
      }
      //if(dev==1){
      //op.map(*this,x0,x1);
      //}
    }

    void map(const BinaryOp<RtensorA,RtensorA,RtensorA>& op, const RtensorA& x0, const RtensorArrayA& x1){
      for(int i=0; i<aasize; i++){
	RtensorA t=cell(i);
	op(t,x0,x1.cell(i));
      }
    }
    
    void map(const BinaryOp<RtensorA,RtensorA,RtensorA>& op, const RtensorArrayA& x0, const RtensorA& x1){
      for(int i=0; i<aasize; i++){
	RtensorA t=cell(i);
	op(t,x0.cell(i),x1);
      }
    }

    void outer(const BinaryOp<RtensorA,RtensorA,RtensorA>& op, const RtensorArrayA& x0, const RtensorArrayA& x1){
      const int I=x0.aasize;
      const int J=x1.aasize;
      assert(aasize=I*J);

      for(int i=0; i<I; i++)
	for(int j=0; j<J; j++){
	  RtensorA t=cell(i*J+j);
	  op(t,x0.cell(i),x1.cell(j));
	}
    }

    void convolve(const BinaryOp<RtensorA,RtensorA,RtensorA>& op, const RtensorArrayA& x0, const RtensorArrayA& x1){
      assert(ak==x0.ak);
      assert(ak==x1.ak);

      if(ak==2){
	const int I=x0.adims[0]-x1.adims[0]+1;
	const int J=x0.adims[1]-x1.adims[1]+1;
	const int cI=x1.adims[0];
	const int cJ=x1.adims[1];
	assert(adims[0]==I);
	assert(adims[1]==J);

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    RtensorA t=cell({i,j});
	    for(int ci=0; ci<cI; ci++)
	      for(int cj=0; cj<cJ; cj++)
		op(t,x0.cell({i+ci,j+cj}),x1.cell({ci,cj}));
	  }
      }
    }


    void matrixprod(const BinaryOp<RtensorA,RtensorA,RtensorA>& op, const RtensorArrayA& x0, const RtensorArrayA& x1){
      assert(ak==2);
      assert(x0.ak==2);
      assert(x1.ak==2);

      if(ak==2){
	const int I=x0.adims[0];
	const int J=x1.adims[1];
	const int K=x0.adims[1];

	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    RtensorA t=cell({i,j});
	    for(int _k=0; _k<K; _k++)
	      op(t,x0.cell({i,_k}),x1.cell({_k,j}));
	  }
      }
    }
    */

    /*
    // deprecated 
    void scatter(const UnaryOp<RtensorA,float>& op, const RtensorA& C){
      for(int i=0; i<aasize; i++){
	RtensorA t=cell(i);
	op(t,C.get_value_at(i));
      }
    }

    // deprecated 
    void scatter(const BinaryOp<RtensorA,RtensorA,float>& op, const RtensorArrayA& x0, const RtensorA& C){
      for(int i=0; i<aasize; i++){
	RtensorA t=cell(i);
	op(t,x0.cell(i),C.get_value_at(i));
      }
    }
    */

    void scatter(const Inplace1Cop<RtensorA,RtensorArrayA,float>& op, const RtensorA& C){
      for(int i=0; i<aasize; i++){
	RtensorA t=cell(i);
	op(t,C.get_value_at(i));
      }
    }

    void scatter(const Unary1Cop<RtensorA,RtensorArrayA,float>& op, const RtensorArrayA& x0, const RtensorA& C){
      for(int i=0; i<aasize; i++){
	RtensorA t=cell(i);
	op(t,x0.cell(i),C.get_value_at(i));
      }
    }


  public: // ---- Map operations -----------------------------------------------------------------------------

    /*
    template<typename OBJ, typename ARR>
    void map(const BinaryCop<OBJ,ARR>& op, const ARR& x0, const ARR& x1){
      if(dev==0){
	for(int i=0; i<aasize; i++){
	  OBJ t=cell(i);
	  op(t,x0.cell(i),x1.cell(i));
	}
      }
      if(dev==1){
	op(*static_cast<ARR*>(this),x0,x1,0);
	//op(*static_cast<ARR*>(this),x0,x1,aasize,1,1, 1,0,0,1,0,0,1,0,0);
      }
    }

    template<typename OBJ, typename ARR>
    void map(const BinaryCop<OBJ,ARR>& op, const OBJ& x, const ARR& y){
      if(dev==0){
	for(int i=0; i<aasize; i++){
	  OBJ t=cell(i);
	  op(t,x,y.cell(i));
	}
      }
      if(dev==1){
	op(*static_cast<ARR*>(this),ARR(x),y,1);
	//op(*static_cast<ARR*>(this),ARR(x0),x1,aasize,1,1, 1,0,0,0,0,0,1,0,0);
      }
    }

    template<typename OBJ, typename ARR>
    void map(const BinaryCop<OBJ,ARR>& op, const ARR& x, const OBJ& y){
      if(dev==0){
	for(int i=0; i<aasize; i++){
	  OBJ t=cell(i);
	  op(t,x.cell(i),y);
	}
      }
      if(dev==1){
	op(*static_cast<ARR*>(this),x,ARR(y),2);
	//op(*static_cast<ARR*>(this),x0,ARR(x1),aasize,1,1, 1,0,0,1,0,0,0,0,0);
      }
    }

    template<typename OBJ, typename ARR>
    void outer(const BinaryCop<OBJ,ARR>& op, const ARR& x, const ARR& y){
      const int I=x.aasize;
      const int J=y.aasize;
      assert(aasize==I*J);

      if(dev==0){
	for(int i=0; i<I; i++)
	  for(int j=0; j<J; j++){
	    OBJ t=cell(i*J+j);
	    op(t,x.cell(i),y.cell(j));
	  }
      }
      if(dev==1){
	op(*static_cast<ARR*>(this),x,y,3);
	//op(*static_cast<ARR*>(this),x,y,I,J,1, dims[1],1,0, 1,0,0, 0,1,0);
      }
    }
    */


  public: // ---- Broadcasting and reductions ----------------------------------------------------------------
    
    
    RtensorArrayA broaden(const int ix, const int n) const{
      assert(ix<=adims.size());
      RtensorArrayA R(*this,adims.insert(ix,n),fill::raw);
      R.broadcast_copy(ix,*this);
      return R;
    }

    RtensorArrayA reduce(const int ix) const{
      assert(ix<adims.size());
      RtensorArrayA R(*this, adims.remove(ix),cnine::fill::zero);
      R.add_reduce(*this,ix);
      return R;
    }

    void broadcast_copy(const RtensorA& x){
      assert(x.dev==dev);
      assert(x.dims==cdims);
      if(dev==0){
	for(int i=0; i<aasize; i++){
	  std::copy(x.arr,x.arr+x.asize,arr+i*cellstride);
	}	
      }else{
	if(aasize==0) return;
	assert(x.asize==asize);
	int s=0;
	int e=1;
	while(e<=aasize){e*=2;s++;}
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,x.asize*sizeof(float),cudaMemcpyDeviceToDevice));  
	e=1; 
	for(int i=0; i<s-1; i++){
	  CUDA_SAFE(cudaMemcpy(arrg+e*cellstride,arrg,e*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  
	  e*=2;
	}
	CUDA_SAFE(cudaMemcpy(arrg+e*cellstride,arrg,(aasize-e)*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }

    void broadcast_copy(const int ix, const RtensorArrayA& x){
      assert(dev==x.dev);
      assert(nbu==x.nbu);
      assert(x.ak==ak-1);
      for(int i=0; i<ix; i++) assert(adims[i]==x.adims[i]);
      for(int i=ix+1; i<adims.size(); i++) assert(adims[i]==x.adims[i-1]);
      int n=adims[ix];
      if(dev==0){
	if(ix==0){
	  for(int i=0; i<n; i++){
	    std::copy(x.arr,x.arr+x.cst,arr+i*strides[0]);
	  }
	}else{
	  const int A=cst/strides[ix-1];
	  for(int a=0; a<A; a++){
	    const int offs=a*strides[ix-1];
	    const int xoffs=a*x.strides[ix-1];
	    for(int i=0; i<n; i++){
	      std::copy(x.arr+xoffs,x.arr+xoffs+x.strides[ix-1],arr+offs+i*strides[ix]);
	    }
	  }
	}
      }

      /*
      if(dev==1){
	if(ix==0){
	  BroadcastCopy_cfloat(arrg,arrgc,x.arrg,x.arrgc,strides[0],n);
	}else{
	  const int A=cst/strides[ix-1];
	  for(int a=0; a<A; a++){
	    const int offs=a*strides[ix-1];
	    const int xoffs=a*x.strides[ix-1];
	    BroadcastCopy_cfloat(arrg+offs,arrgc+offs,x.arrg+xoffs,x.arrgc+xoffs,strides[ix],n);
	  }
	}
      }
      */
    }


    void broadcast_add(const RtensorA& x){
      CNINE_CHECK_SIZE(cdims.check_cell_eq(x.dims));
      assert(dev==x.dev);
      assert(nbu==x.nbu);
      //assert(x.cdims==cdims);
      if(dev==0){
	for(int i=0; i<aasize; i++){
	  stdadd(x.arr,x.arr+x.asize,arr+i*cellstride);
	}
      }
    }


    void broadcast_subtract(const RtensorA& x){
      CNINE_CHECK_SIZE(cdims.check_cell_eq(x.dims));
      assert(dev==x.dev);
      assert(nbu==x.nbu);
      //assert(x.cdims==cdims);
      if(dev==0){
	for(int i=0; i<aasize; i++){
	  stdsub(x.arr,x.arr+x.asize,arr+i*cellstride);
	}
      }
    }


    void broadcast_add(const int ix, const RtensorArrayA& x){
      assert(dev==x.dev);
      assert(nbu==x.nbu);
      assert(x.ak==ak-1);
      for(int i=0; i<ix; i++) assert(adims[i]==x.adims[i]);
      for(int i=ix+1; i<adims.size(); i++) assert(adims[i]==x.adims[i-1]);
      int n=adims[ix];
      if(dev==0){
	if(ix==0){
	  for(int i=0; i<n; i++){
	    stdadd(x.arr,x.arr+x.cst,arr+i*strides[0]);
	  }
	}else{
	  const int A=cst/strides[ix-1];
	  for(int a=0; a<A; a++){
	    const int offs=a*strides[ix-1];
	    const int xoffs=a*x.strides[ix-1];
	    for(int i=0; i<n; i++){
	      stdadd(x.arr+xoffs,x.arr+xoffs+x.strides[ix-1],arr+offs+i*strides[ix]);
	    }
	  }
	}
      }

      /*
      if(dev==1){
	if(ix==0)
	  BroadcastAdd_cfloat(arrg,arrgc,x.arrg,x.arrgc,strides[0],n);
	else{
	  const int A=cst/strides[ix-1];
	  for(int a=0; a<A; a++){
	    const int offs=a*strides[ix-1];
	    const int xoffs=a*x.strides[ix-1];
	    BroadcastAdd_cfloat(arrg+offs,arrgc+offs,x.arrg+xoffs,x.arrgc+xoffs,strides[ix],n);
	  }
	}
      }
      */

    }


    void add_reduce(const RtensorArrayA& x, const int ix){
      assert(dev==x.dev);
      assert(nbu==x.nbu);
      assert(ak==x.ak-1 || x.ak==1);
      for(int i=0; i<ix; i++) assert(adims[i]==x.adims[i]);
      for(int i=ix+1; i<adims.size(); i++) assert(x.adims[i]==adims[i-1]);
      int n=x.adims[ix];

      if(dev==0){
	if(ix==0){
	  for(int i=0; i<n; i++){
	    stdadd(x.arr+i*x.strides[0],x.arr+(i+1)*x.strides[0],arr);
	  }
	}else{
	  const int A=x.cst/x.strides[ix-1];
	  for(int a=0; a<A; a++){
	    const int xoffs=a*x.strides[ix-1];
	    const int offs=a*strides[ix-1];
	    for(int i=0; i<n; i++){
	      stdadd(x.arr+xoffs+i*x.strides[ix],x.arr+xoffs+(i+1)*x.strides[ix],arr+offs);
	    }
	  }	  
	}
      }

      /*
      if(dev==1){
	if(ix==0)
	  ReduceAdd_cfloat(arrg,arrgc,x.arrg,x.arrgc,strides[0],n);
	else{
	  const int A=x.cst/x.strides[ix-1];
	  for(int a=0; a<A; a++){
	    const int xoffs=a*x.strides[ix-1];
	    const int offs=a*strides[ix-1];
	    ReduceAdd_cfloat(arrg+offs,arrgc+offs,x.arrg+xoffs,x.arrgc+xoffs,strides[ix],n);
	  }	  
	}
      }
      */

    }


  public: // ---- Multiplication by scattered matrices --------------------------------------------------------


    void scatter_add_times_c(const RtensorArrayA& x, const RtensorA& C){
      assert(adims==x.dims);
      RtensorA_add_times_c_cop op;
      scatter(op,x,C);
    }

    void scatter_add_div_c(const RtensorArrayA& x, const RtensorA& C){
      assert(adims==x.dims);
      RtensorA_add_div_c_cop op;
      scatter(op,x,C);
    }

    void inplace_scatter_times(const RtensorA& C){
      assert(adims==C.dims);
      /*
	if(dev==0){
	assert(C.dev==0);
	for(int i=0; i<aasize; i++){
	float c=float(C.arr[i],C.arrc[i]);
	auto acc=accessor(i);
	for(int j=0; j<asize; j++)
	acc[j]*=c;
	}
      */
      RtensorA_inplace_times_c_cop op;
      scatter(op,C);
    }

    void inplace_scatter_div(const RtensorA& C){
      assert(adims==C.dims);
      /*
      if(dev==0){
	assert(C.dev==0);
	for(int i=0; i<aasize; i++){
	  float c=float(C.arr[i],C.arrc[i]);
	  auto acc=accessor(i);
	  for(int j=0; j<asize; j++)
	    acc[j]/=c;
	}
      }
      */
      RtensorA_inplace_div_c_cop op;
      scatter(op,C);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    RtensorArrayA plus(const RtensorArrayA& y) const{
      RtensorArrayA R(*this);
      R.add(y);
      return R;
    }

    RtensorArrayA plus(const RtensorA& y) const{
      RtensorArrayA R(*this);
      R.broadcast_add(y);
      return R;
    }

    RtensorArrayA minus(const RtensorA& y) const{
      RtensorArrayA R(*this);
      R.broadcast_subtract(y);
      return R;
    }

    
  public: // ---- Cellwise cumulative operations -------------------------------------------------------------


    void add_Mprod_AA(const RtensorArrayA& x, const RtensorArrayA& y, const int nx=1, const int ny=1){
      RtensorA_add_Mprod_AA_cop op(nx,ny);
      CellwiseBiCmap(op,*this,x,y);
    }

    void add_Mprod_AT(const RtensorArrayA& x, const RtensorArrayA& y, const int nx=1, const int ny=1){
      RtensorA_add_Mprod_AT_cop op(nx,ny);
      CellwiseBiCmap(op,*this,x,y);
    }


  public: // ---- Broadcast cumulative operations ------------------------------------------------------------


    void add_broadcast(const RtensorA& y){
      RtensorA_add_cop op;
      BroadcastUCmap(op,*this,RtensorArrayA(y));
    }

    void subtract_broadcast(const RtensorA& y){
      RtensorA_subtract_cop op;
      BroadcastUCmap(op,*this,RtensorArrayA(y));
    }

    void broadcast_add_Mprod_AA(const RtensorArrayA& x, const RtensorA& y, const int nx=1, const int ny=1){
      assert(x.dev==dev);
      assert(y.dev==dev);
      if(x.asize==0 || y.asize==0) return;

      const int K=x.cell_combined_size(x.k-nx,x.k);
      assert(y.combined_size(0,ny)==K);
      const int I=x.cell_combined_size(0,x.k-nx);
      const int J=y.combined_size(ny,y.k);
      assert(asize==I*J);
      if(asize==0) return;

      if(dev==0){

	const int istridex=K;
	const int istrider=J;
	const int pstridey=J;

	for(int a=0; a<aasize; a++){
	  auto accr=accessor(a);
	  auto accx=const_cast<RtensorArrayA&>(x).accessor(a);
	  auto accy=const_cast<RtensorA&>(y).accessor();
	  for(int i=0; i<I; i++)
	    for(int j=0; j<J; j++){
	      float t=0; 
	      for(int p=0; p<K; p++){
		//cout<<i<<" "<<j<<" "<<p<<endl;
		int qx=i*istridex+p;
		int qy=p*pstridey+j;
		t+=accx[qx]*accy[qy];
	      }
	      int qr=i*istrider+j;
	      accr[qr]+=t; 
	    }
	}
      }

      if(dev>0){

	float alpha0=1.0;
	float beta=1.0;
	
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_N,CUBLAS_OP_N,J,I,K,&alpha0,
	    y.arrg,J,x.arrg,K,&beta,arrg,J)); 
	//cudaDeviceSynchronize(); 
      }

    }


    // The last nx indices of x are contracted with the last ny indices of y
    void broadcast_add_Mprod_AT(const RtensorArrayA& x, const RtensorA& y, const int nx=1, const int ny=1){
      assert(x.dev==dev);
      assert(y.dev==dev);
      if(x.asize==0 || y.asize==0) return;

      const int K=x.cell_combined_size(x.k-nx,x.k);
      assert(y.combined_size(y.k-ny,y.k)==K);
      const int I=x.cell_combined_size(0,x.k-nx);
      const int J=y.combined_size(0,y.k-ny);
      assert(asize==I*J);
      if(asize==0) return;

      if(dev==0){
	assert(x.dev==0);
	assert(y.dev==0);

	const int istridex=K;
	const int istrider=J;
	const int jstridey=K;

	for(int a=0; a<aasize; a++){
	  auto accr=accessor(a);
	  auto accx=const_cast<RtensorArrayA&>(x).accessor(a);
	  auto accy=const_cast<RtensorA&>(y).accessor();

	  for(int i=0; i<I; i++)
	    for(int j=0; j<J; j++){
	      float t=0; 
	      for(int p=0; p<K; p++){
		int qx=i*istridex+p;
		int qy=p+j*jstridey;
		t+=accx[qx]*accy[qy];
	      }
	      int qr=i*istrider+j;
	      accr[qr]+=t; 
	    }
	  
	}

      }

      if(dev>0){

	float alpha0=1.0;
	float beta=1.0;
	
	CUBLAS_SAFE(cublasSgemm(cnine_cublas,CUBLAS_OP_T,CUBLAS_OP_N,J,I,K,&alpha0,
	    y.arrg,K,x.arrg,K,&beta,arrg,J)); 
	//cudaDeviceSynchronize(); 
      }

    }


  public: // ---- Scatter operations -------------------------------------------------------------------------


    //void scatter_add_times_c(const RtensorArrayA& x){
    //}


  public: // ---- ConvolveInterpolate ------------------------------------------------------------------------




  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;

      for(int i=0; i<aasize; i++){
	Gindex aix(i,adims);
	oss<<indent<<"Cell "<<aix<<endl;
	oss<<get_cell(aix).str(indent)<<endl<<endl;
      }

      return oss.str();

    }
    
    friend ostream& operator<<(ostream& stream, const RtensorArrayA& x){
      stream<<x.str(); return stream;}

  };

}


#endif 
