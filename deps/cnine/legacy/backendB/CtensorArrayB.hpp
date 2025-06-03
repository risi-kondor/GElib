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


#ifndef _CnineCtensorArrayB
#define _CnineCtensorArrayB

#include "CtensorB.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{


  class CtensorArrayB: public CtensorB{
  public:

    int ak;
    bool batched=false;

    CtensorArrayB(){}

    ~CtensorArrayB(){
    }

    string classname() const{
      return "CtensorArrayB";
    }

    string describe() const{
      return "CtensorArrayB"+dims.str();
    }


  public: // ---- Constructors -----------------------------------------------------------------------------


    CtensorArrayB(const Gdims& _adims, const Gdims& _dims, const int _dev=0): 
      CtensorB(Gdims(_adims,_dims),_dev), ak(_adims.size()){}


  public: // ---- Filled constructors -----------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorArrayB(const Gdims& _adims, const Gdims& _dims, const FILLTYPE& dummy, const int _dev=0): 
      CtensorB(Gdims(_adims,_dims),dummy,_dev), ak(_adims.size()){}

    
  public: // ---- Named constructors -------------------------------------------------------------------------


    static CtensorArrayB noalloc(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArrayB(_adims,_dims,fill_noalloc(),_dev);
    }

    static CtensorArrayB raw(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArrayB(_adims,_dims,fill_raw(),_dev);
    }

    static CtensorArrayB zero(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArrayB(_adims,_dims,fill_zero(),_dev);
    }

    static CtensorArrayB zeros(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArrayB(_adims,_dims,fill_zero(),_dev);
    }

    static CtensorArrayB ones(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArrayB(_adims,_dims,fill_ones(),_dev);
    }

    static CtensorArrayB sequential(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArrayB(_adims,_dims,fill_sequential(),_dev);
    }

    static CtensorArrayB gaussian(const Gdims& _adims, const Gdims& _dims, const int _dev=0){
      return CtensorArrayB(_adims,_dims,fill_gaussian(),_dev);
    }


    static CtensorArrayB raw_like(const CtensorArrayB& x){
      return CtensorArrayB(x.get_adims(),x.get_cdims(),fill_raw(),x.dev);
    }

    static CtensorArrayB raw_like(const CtensorB& x, const Gdims& _adims){
      return CtensorArrayB(_adims,x.dims,fill_raw(),x.dev);
    }

    static CtensorArrayB raw_like(const CtensorArrayB& x, const Gdims& _adims){
      return CtensorArrayB(_adims,x.get_cdims(),fill_raw(),x.dev);
    }

    static CtensorArrayB zeros_like(const CtensorArrayB& x){
      return CtensorArrayB(x.get_adims(),x.get_cdims(),fill_zero(),x.dev);
    }

    static CtensorArrayB zeros_like(const CtensorArrayB& x, const Gdims& _adims){
      return CtensorArrayB(_adims,x.get_cdims(),fill_zero(),x.dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    CtensorArrayB(const CtensorArrayB& x):
      CtensorB(x), ak(x.ak), batched(x.batched){}

    CtensorArrayB(const CtensorArrayB& x, const view_flag& dummy):
      CtensorB(x,dummy), ak(x.ak), batched(x.batched){}

    CtensorArrayB(CtensorArrayB&& x):
      CtensorB(std::move(x)), ak(x.ak), batched(x.batched){}

    CtensorArrayB& operator=(const CtensorArrayB& x){
      CtensorB::operator=(x);
      ak=x.ak;
      batched=x.batched;
      return *this;
    }

    CtensorArrayB& operator=(CtensorArrayB&& x){
      CtensorB::operator=(std::move(x));
      ak=x.ak;
      batched=x.batched;
      return *this;
    }


  public: // ---- Views -------------------------------------------------------------------------------------


    CtensorArrayB view(){
      return CtensorArrayB(CtensorB::view(),ak);
    }
    
    CtensorArrayB view_as_shape(const Gdims& _adims){
      CNINE_DIMS_EQ_TOTAL(get_adims(),_adims)
      CtensorArrayB R=CtensorArrayB::noalloc(_adims,get_cdims(),dev);
      R.arr=arr;
      R.arrg=arrg;
      R.is_view=true;
      return R;
    }

    CtensorView fuse_array_indices(){
      int c=get_ncdims();
      Gdims _dims(c+1,fill_raw());
      int t=1;
      for(int i=0; i<ak; i++) 
	t*=dims[i];
      _dims[0]=t;
      for(int i=0; i<c; i++)
	_dims[i+1]=dims[i+ak];
      return CtensorView(arr,arr+coffs,_dims,strides.chunk(ak-1));
    }

    const CtensorView fuse_array_indices() const{
      int c=get_ncdims();
      Gdims _dims(c+1,fill_raw());
      int t=1;
      for(int i=0; i<ak; i++) 
	t*=dims[i];
      _dims[0]=t;
      for(int i=0; i<c; i++)
	_dims[i+1]=dims[i+ak];
      return CtensorView(arr,arr+coffs,_dims,strides.chunk(ak-1));
    }



  public: // ---- Conversions -------------------------------------------------------------------------------


    CtensorArrayB(const CtensorB& x, const int _ak, const bool _batched=false):
      CtensorB(x), batched(_batched){
      if(_ak<0) ak=dims.size()+_ak;
      else ak=_ak;
    }

    CtensorArrayB(CtensorB&& x, const int _ak, const bool _batched=false):
      CtensorB(std::move(x)), batched(_batched){
      if(_ak<0) ak=dims.size()+_ak;
      else ak=_ak;
      is_view=x.is_view;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    CtensorArrayB(const int _ak, const at::Tensor& T, const bool _batched=false):
      CtensorArrayB(CtensorB(T),_ak,_batched){}

    CtensorArrayB(const at::Tensor& T, const int _ak, const bool _batched=false):
      CtensorArrayB(CtensorB(T),_ak,_batched){}

    static CtensorArrayB view(at::Tensor& T, const int _ak, const bool _batched=false){
      //cout<<"Batched="<<_batched<<endl;
      return CtensorArrayB(CtensorB::view(T),_ak,_batched);
    }

    static CtensorArrayB* viewp(at::Tensor& T, const int _ak, const bool _batched=false){
      return new CtensorArrayB(CtensorB::view(T),_ak,_batched);
    }

#endif 


  public: // ---- Access -------------------------------------------------------------------------------------


    Gdims get_adims() const{
      return dims.chunk(0,ak);
    }

    int get_nadims() const{
      return ak;
    }

    int get_adim(const int i) const{
      return dims[i];
    }

    int get_aasize() const{
      return get_adims().asize();
    }


    Gdims get_cdims() const{
      return dims.chunk(ak);
    }

    int get_ncdims() const{
      return dims.size()-ak;
    }

    int get_cdim(const int i) const{
      return dims[ak+i];
    }

    int get_casize() const{
      return get_cdims().asize();
    }


    int get_cellstride() const{
      return strides[ak];
    }


  public: // ---- Element Access -----------------------------------------------------------------------------


    complex<float> operator()(const Gindex& aix, const Gindex& cix) const{
      return CtensorB::operator()(aix.cat(cix));
    }

    void set(const Gindex& aix, const Gindex& cix, complex<float> v) const{
      return CtensorB::set(aix.cat(cix),v);
    }


  public: // ---- Cell Access --------------------------------------------------------------------------------


    CtensorB get_cell(const Gindex& aix) const{
      CtensorB R(get_cdims(),fill::raw,dev);
      copy_cell_to(R,aix);
      return R;
    }

    void copy_cell_to(CtensorB& R, const Gindex& aix) const{
      CNINE_DEVICE_SAME(R);
      CNINE_CHECK_RANGE(aix.check_arange(get_adims()));
      int t=aix(strides);
      if(dev==0) std::copy(arr+t,arr+t+R.memsize,R.arr);
      if(dev==1) CUDA_SAFE(cudaMemcpy(R.arrg,arrg+t,R.memsize*sizeof(float),cudaMemcpyDeviceToDevice));
    }

    void add_cell_to(CtensorB& R, const Gindex& aix) const{
      CNINE_DEVICE_SAME(R);
      CNINE_CHECK_RANGE(aix.check_arange(get_adims()));
      int t=aix(strides);
      if(dev==0) stdadd(arr+t,arr+t+R.memsize,R.arr);
      if(dev==1){
	const float alpha=1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, R.memsize, &alpha, arrg+t, 1, R.arrg, 1));
      }
    }


    CtensorB get_cell(const int ix) const{
      CtensorB R(get_cdims(),fill::raw,dev);
      copy_cell_to(R,ix);
      return R;
    }

    void copy_cell_to(CtensorB& R, const int ix) const{
      CNINE_DEVICE_SAME(R);
      int t=ix*get_cellstride();
      if(dev==0) std::copy(arr+t,arr+t+R.memsize,R.arr);
      if(dev==1) CUDA_SAFE(cudaMemcpy(R.arrg,arrg+t,R.memsize*sizeof(float),cudaMemcpyDeviceToDevice));
    }

    void add_cell_to(CtensorB& R, const int ix) const{
      CNINE_DEVICE_SAME(R);
      int t=ix*get_cellstride();
      if(dev==0) stdadd(arr+t,arr+t+R.memsize,R.arr);
      if(dev==1){
	const float alpha=1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, R.memsize, &alpha, arrg+t, 1, R.arrg, 1));
      }
    }


    void set_cell(const Gindex& aix, const CtensorB& x) const{
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_RANGE(aix.check_arange(get_adims()));
      int t=aix(strides);
      if(dev==0) std::copy(x.arr,x.arr+x.memsize,arr+t);
      if(dev==1) CUDA_SAFE(cudaMemcpy(arrg+t,x.arrg,x.memsize*sizeof(float),cudaMemcpyDeviceToDevice));
    }

    void add_to_cell(const Gindex& aix, const CtensorB& x) const{
      CNINE_DEVICE_SAME(x);
      CNINE_CHECK_RANGE(aix.check_arange(get_adims()));
      int t=aix(strides);
      if(dev==0) stdadd(x.arr,x.arr+x.memsize,arr+t);
      if(dev==1){
	const float alpha=1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, x.memsize, &alpha, x.arrg, 1, arrg+t, 1));
      }
    }

    void set_cell(const int ix, const CtensorB& x) const{
      CNINE_DEVICE_SAME(x);
      int t=ix*strides[ak];
      if(dev==0) std::copy(x.arr,x.arr+x.memsize,arr+t);
      if(dev==1) CUDA_SAFE(cudaMemcpy(arrg+t,x.arrg,x.memsize*sizeof(float),cudaMemcpyDeviceToDevice));
    }

    void add_to_cell(const int ix, const CtensorB& x) const{
      CNINE_DEVICE_SAME(x);
      int t=ix*strides[ak];
      if(dev==0) stdadd(x.arr,x.arr+x.memsize,arr+t);
      if(dev==1){
	const float alpha=1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, x.memsize, &alpha, x.arrg, 1, arrg+t, 1));
      }
    }


    CtensorB cell(const int i){
      return CtensorB::view_of_array(get_cdims(),arr+i*get_cellstride(),dev);
    }

    CtensorB cell(const int i) const{
      return CtensorB::view_of_array(get_cdims(),arr+i*get_cellstride(),dev);
    }

    CtensorB cell(const int i, const int j){
      return CtensorB::view_of_array(get_cdims(),arr+i*strides[0]+j*strides[1],dev);
    }

    CtensorB cell(const int i, const int j) const{
      return CtensorB::view_of_array(get_cdims(),arr+i*strides[0]+j*strides[1],dev);
    }

    CtensorB cell(const Gindex& aix){
      CNINE_CHECK_RANGE(aix.check_arange(get_adims()));
      return CtensorB::view_of_array(get_cdims(),arr+aix(strides),dev);
    }

    CtensorB cell(const Gindex& aix) const{
      CNINE_CHECK_RANGE(aix.check_arange(get_adims()));
      return CtensorB::view_of_array(get_cdims(),arr+aix(strides),dev);
    }


    /*
    CtensorB cell(const int i, const int j){
      //cout<<"CtensorB view"<<endl;
      return CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+(i*astrides[0]+j*astrides[1])*cellstride,
	arrc+(i*astrides[0]+j*astrides[1])*cellstride,flag::view);
    }

    const CtensorB cell(const int i, const int j) const{
      //cout<<"const CtensorB view"<<endl;
      return CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+(i*astrides[0]+j*astrides[1])*cellstride,
	arrc+(i*astrides[0]+j*astrides[1])*cellstride,flag::view);
    }

    CtensorB cell_view(const int i, const int j){
      //cout<<"CtensorB view"<<endl;
      return CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+(i*astrides[0]+j*astrides[1])*cellstride,
	arrc+(i*astrides[0]+j*astrides[1])*cellstride,flag::view);
    }

    CtensorB cell(const Gindex& aix){
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      //cout<<"CtensorB view"<<endl;
      int i=aix(astrides);
      return CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,arrc+i*cellstride,flag::view);
    }

    const CtensorB cell(const Gindex& aix) const{
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      //cout<<"const CtensorB view"<<endl;
      int i=aix(astrides);
      return CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,arrc+i*cellstride,flag::view);
    }

    CtensorB cell_view(const Gindex& aix){
      CNINE_CHECK_RANGE(aix.check_arange(adims));
      //cout<<"CtensorB view"<<endl;
      int i=aix(astrides);
      return CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,arrc+i*cellstride,flag::view);
    }


    CtensorB* cellp(const int i){
      return new CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,arrc+i*cellstride,flag::view);
    }

    const CtensorB* cellp(const int i) const{
      return new CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,arrc+i*cellstride,flag::view);
    }

    CtensorB* cellp(const Gindex& aix){
      int i=aix(astrides);
      return new CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,arrc+i*cellstride,flag::view);
    }

    const CtensorB* cellp(const Gindex& aix) const{
      int i=aix(astrides);
      return new CtensorB(k,cdims,nbu,cstrides,asize,2*asize,dev,arr+i*cellstride,arrc+i*cellstride,flag::view);
    }
    */


    //CtensorB_accessor accessor(const int i){
    //return CtensorB_accessor(arr+i*cellstride,arrc+i*cellstride,cstrides);
    //}

    //CtensorB_accessor accessor(const int i) const{
    //return CtensorB_accessor(arr+i*cellstride,arrc+i*cellstride,cstrides);
    //}

    //CtensorA_spec cellspec() const{
    //return CtensorA_spec(cdims,nbu,cstrides,asize);
    //}




  public: // ---- Operations ---------------------------------------------------------------------------------


    CtensorArrayB gather(const Rmask1& mask){
      Gdims _dims(dims);
      _dims[0]=mask.N0;
      CtensorArrayB R(CtensorB::zero(_dims,dev),ak);
      R.add_gather(*this,mask);
      return R;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_gather(const CtensorB& x, const Rmask1& mask){
      cout<<"XXXXXX"<<endl;
      if(ak!=1) throw std::invalid_argument("CtensorArrayB::add_gather(const CtensorB&, const Rmask1&): number of array arguments must be 1.");
      assert(x.dims.size()==dims.size());
      Aggregator(viewx(),x.viewx(),mask);
    }
    

  public: // ---- Broadcasting and reductions ----------------------------------------------------------------


    void broadcast_copy(const CtensorB& x){
      CNINE_DEVICE_EQ(x,(*this));
      assert(x.dims==get_cdims());
      int aasize=get_aasize();
      int cellstride=get_cellstride();
      if(dev==0){
	for(int i=0; i<aasize; i++){
	  std::copy(x.arr,x.arr+x.memsize,arr+i*cellstride);
	}	
      }else{
	if(aasize==0) return;
	int s=0;
	int e=1;
	while(e<=aasize){e*=2;s++;}
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,x.memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	e=1; 
	for(int i=0; i<s-1; i++){
	  CUDA_SAFE(cudaMemcpy(arrg+e*cellstride,arrg,e*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  
	  e*=2;
	}
	CUDA_SAFE(cudaMemcpy(arrg+e*cellstride,arrg,(aasize-e)*cellstride*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }


    void broadcast_copy(const int ix, const CtensorArrayB& x){
      CNINE_DEVICE_EQ(x,(*this));
      assert(x.ak==ak-1);

      if([&](const Gdims& adims, const Gdims& xadims){
	  for(int i=0; i<ix; i++) if(adims[i]!=xadims[i]) return true;
	  for(int i=ix+1; i<adims.size(); i++) if(adims[i]!=xadims[i-1]) return true;
	  return false;
	}(get_adims(),x.get_adims())) 
	throw std::out_of_range("Cnine error in "+string(__PRETTY_FUNCTION__)+": dimension mismatch");
      
      int n=get_adims()[ix];
      if(dev==0){
	if(ix==0){
	  for(int i=0; i<n; i++){
	    std::copy(x.arr,x.arr+x.memsize,arr+i*strides[0]);
	  }
	}else{
	  const int A=memsize/strides[ix-1];
	  for(int a=0; a<A; a++){
	    const int offs=a*strides[ix-1];
	    const int xoffs=a*x.strides[ix-1];
	    for(int i=0; i<n; i++){
	      std::copy(x.arr+xoffs,x.arr+xoffs+x.strides[ix-1],arr+offs+i*strides[ix]);
	    }
	  }
	}
      }

      if(dev==1){
	CNINE_CPUONLY();
	/*
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
	*/
      }
    }


    void add_reduce(const CtensorArrayB& x, const int ix){
      CNINE_DEVICE_EQ(x,(*this));
      assert(ak==x.ak-1 || x.ak==1);
      if([&](const Gdims& adims, const Gdims& xadims){
	  for(int i=0; i<ix; i++) if(adims[i]!=adims[i]) return true;
	  for(int i=ix+1; i<adims.size(); i++) if(xadims[i]!=adims[i-1]) return true;
	  return false;
	}(get_adims(), x.get_adims()))
	throw std::out_of_range("Cnine error in "+string(__PRETTY_FUNCTION__)+": dimension mismatch");
 
      int n=x.get_adims()[ix];

      if(dev==0){
	if(ix==0){
	  for(int i=0; i<n; i++){
	    stdadd(x.arr+i*x.strides[0],x.arr+(i+1)*x.strides[0],arr);
	  }
	}else{
	  const int A=x.memsize/x.strides[ix-1];
	  for(int a=0; a<A; a++){
	    const int xoffs=a*x.strides[ix-1];
	    const int offs=a*strides[ix-1];
	    for(int i=0; i<n; i++){
	      stdadd(x.arr+xoffs+i*x.strides[ix],x.arr+xoffs+(i+1)*x.strides[ix],arr+offs);
	    }
	  }	  
	}
      }

      if(dev==1){
	CNINE_CPUONLY();
	/*
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
	*/
      }

    }


  public: // ---- Multiplication by scattered matrices --------------------------------------------------------


    void scatter_add_times_c(const CtensorArrayB& x, const CtensorB& C){
      //assert(adims==x.dims);
      //CtensorA_add_times_c_cop op;
      //scatter(op,x,C);
    }

    void scatter_add_div_c(const CtensorArrayB& x, const CtensorB& C){
      //assert(adims==x.dims);
      //CtensorA_add_div_c_cop op;
      //scatter(op,x,C);
    }

    void inplace_scatter_times(const CtensorB& C){
      //assert(adims==C.dims);
      /*
	if(dev==0){
	assert(C.dev==0);
	for(int i=0; i<aasize; i++){
	complex<float> c=complex<float>(C.arr[i],C.arrc[i]);
	auto acc=accessor(i);
	for(int j=0; j<asize; j++)
	acc[j]*=c;
	}
      */
      //CtensorA_inplace_times_c_cop op;
      //scatter(op,C);
    }

    void inplace_scatter_div(const CtensorB& C){
      //assert(adims==C.dims);
      /*
      if(dev==0){
	assert(C.dev==0);
	for(int i=0; i<aasize; i++){
	  complex<float> c=complex<float>(C.arr[i],C.arrc[i]);
	  auto acc=accessor(i);
	  for(int j=0; j<asize; j++)
	    acc[j]/=c;
	}
      }
      */
      //CtensorA_inplace_div_c_cop op;
      //scatter(op,C);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    CtensorArrayB conj() const{
      return CtensorArrayB(CtensorB::conj(),ak);
    }

    CtensorArrayB plus(const CtensorArrayB& x) const{
      return CtensorArrayB(CtensorB::plus(x),ak);
    }

    CtensorArrayB plus(const CtensorB& y) const{
      CtensorArrayB R(*this);
      //R.broadcast_add(y);
      return R;
    }

    CtensorArrayB minus(const CtensorB& y) const{
      CtensorArrayB R(*this);
      //R.broadcast_subtract(y);
      return R;
    }


  public: // ---- ConvolveInterpolate ------------------------------------------------------------------------
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      assert(ak<dims.size());
      Gdims arraydims=dims.chunk(0,ak);
      arraydims.foreach_index([&](const vector<int>& ix){
	  oss<<indent<<"Cell"<<Gindex(ix)<<endl;
	  oss<<get_cell(ix).str(indent)<<endl;
	});
      return oss.str();
    }

    string repr() const{
      return "<cnine::CtensorArrayB"+dims.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const CtensorArrayB& x){
      stream<<x.str(); return stream;}
   




    

  };

}

#endif 
