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


#ifndef _CnineIntTensor
#define _CnineIntTensor

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"
#include "Itensor3_view.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{


  class IntTensor{
  public:

    Gdims dims;
    Gstrides strides;
    int memsize=0;
    int dev=0;
    bool is_view=false;

    int* arr=nullptr;
    int* arrg=nullptr;

    ~IntTensor(){
      if(is_view) return;
      if(arr) {delete[] arr;}
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }

    string classname() const{
      return "IntTensor";
    }

    string describe() const{
      return "IntTensor"+dims.str();
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    IntTensor(){}

    IntTensor(const Gdims& _dims, const Gstrides& _strides, const int _dev=0): 
      dims(_dims), strides(_strides), dev(_dev){}

    IntTensor(const Gdims& _dims, const int _dev=0): 
      dims(_dims), dev(_dev), strides(_dims){
      CNINE_DIMS_VALID(dims);
      CNINE_DEVICE_VALID(dev);
      memsize=strides[0]*dims[0]; 

      if(dev==0){
	arr=new int[std::max(memsize,1)];
      }

      if(dev==1){
	cout<<1221212<<endl;
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
      }

    }


  public: // ---- Filled constructors -----------------------------------------------------------------------

    
    IntTensor(const Gdims& _dims, const fill_noalloc& dummy, const int _dev=0): 
      dims(_dims), dev(_dev), strides(_dims){}
    
    IntTensor(const Gdims& _dims, const fill_raw& dummy, const int _dev=0): 
      IntTensor(_dims,_dev){}
    
    IntTensor(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      IntTensor(_dims,_dev){
      int asize=strides[0]*dims[0]; 
      CPUCODE(std::fill(arr,arr+asize,0));
      GPUCODE(CUDA_SAFE(cudaMemset(arrg,0,asize*sizeof(int))));
    }

    IntTensor(const Gdims& _dims, const fill_ones& dummy, const int _dev=0): 
      IntTensor(_dims,_dev){
      int asize=strides[0]*dims[0]; 
      CPUCODE(std::fill(arr,arr+asize,1));
      GPUCODE(CNINE_CPUONLY());
    }

    IntTensor(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0): 
      IntTensor(_dims,_dev){
      int asize=strides[0]*dims[0]; 
      CPUCODE(for(int i=0; i<asize; i++) arr[i]=i;);
      GPUCODE(CNINE_CPUONLY());
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static IntTensor noalloc(const Gdims& _dims, const int _dev=0){
      return IntTensor(_dims,fill_noalloc(),_dev);
    }

    static IntTensor raw(const Gdims& _dims, const int _dev=0){
      return IntTensor(_dims,fill_raw(),_dev);
    }

    static IntTensor zero(const Gdims& _dims, const int _dev=0){
      return IntTensor(_dims,fill_zero(),_dev);
    }

    static IntTensor zeros(const Gdims& _dims, const int _dev=0){
      return IntTensor(_dims,fill_zero(),_dev);
    }

    static IntTensor ones(const Gdims& _dims, const int _dev=0){
      return IntTensor(_dims,fill_ones(),_dev);
    }

    static IntTensor sequential(const Gdims& _dims, const int _dev=0){
      return IntTensor(_dims,fill_sequential(),_dev);
    }


    template<typename FILL>
    static IntTensor like(const IntTensor& x){
      return IntTensor(x.dims,FILL(),x.dev);
    }

    static IntTensor raw_like(const IntTensor& x){
      return IntTensor(x.dims,fill_raw(),x.dev);
    }

    static IntTensor zeros_like(const IntTensor& x){
      return IntTensor(x.dims,fill_zero(),x.dev);
    }


  public: // ---- Initializing constructors -----------------------------------------------------------------

    
    IntTensor(const vector<int>& v):
      IntTensor(Gdims(v.size()),fill_raw()){
      for(int i=0; i<v.size(); i++)
	set(i,v[i]);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    IntTensor(const IntTensor& x): 
      IntTensor(x.dims,x.strides,x.dev){
      CNINE_COPY_WARNING();
      memsize=strides[0]*dims[0];
      if(dev==0){
	arr=new int[std::max(memsize,1)];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	cout<<1321232<<endl;
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(int),cudaMemcpyDeviceToDevice));
      }
    }
        
    IntTensor(const IntTensor& x, const nowarn_flag& dummy): 
      IntTensor(x.dims,x.strides,x.dev){
      memsize=strides[0]*dims[0]; 
      if(dev==0){
	arr=new int[std::max(memsize,1)];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	cout<<1321232<<endl;
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(int),cudaMemcpyDeviceToDevice));
      }
    }
        
    IntTensor(const IntTensor& x, const int _dev): 
      IntTensor(x.dims,x.strides,_dev){
      memsize=strides[0]*dims[0]; 
      if(memsize==0) memsize=1;
      if(dev==0){
	if(x.dev==0){
	  arr=new int[std::max(memsize,1)];
	  std::copy(x.arr,x.arr+memsize,arr);
	}
	if(x.dev==1){
	  CNINE_REQUIRES_CUDA();
	  arr=new int[std::max(memsize,1)];
	  CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(int),cudaMemcpyDeviceToHost)); 
	}
      }
      if(dev==1){
	CNINE_REQUIRES_CUDA();
	cout<<1321232<<endl;
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
	if(x.dev==0){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(int),cudaMemcpyHostToDevice));
	}
	if(x.dev==1){
	  CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(int),cudaMemcpyDeviceToDevice));  
	}
      }
    }

    IntTensor(const IntTensor& x, const view_flag& dummy):
      IntTensor(x.dims,x.strides,x.dev){
      arr=x.arr;
      arrg=x.arrg;
      memsize=x.memsize;
      is_view=true;
    }
        
    IntTensor(IntTensor&& x): 
      IntTensor(x.dims,x.strides,x.dev){
      CNINE_MOVE_WARNING();
      arr=x.arr; x.arr=nullptr; 
      arrg=x.arrg; x.arrg=nullptr;
      memsize=x.memsize; x.memsize=0;
      is_view=x.is_view;
    }

    IntTensor& operator=(const IntTensor& x){
      CNINE_ASSIGN_WARNING();
      if(!is_view && arr) {delete[] arr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
      dims=x.dims;
      strides=x.strides;
      dev=x.dev;
      memsize=strides[0]*dims[0]; 
      is_view=false;
      if(dev==0){
	arr=new int[std::max(memsize,1)];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	cout<<1321232<<endl;
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(int),cudaMemcpyDeviceToDevice));
      }
      return *this;
    }

    IntTensor& operator=(IntTensor&& x){
      CNINE_MOVEASSIGN_WARNING();
      if(!is_view && arr) {delete[] arr;}
      if(!is_view && arrg) {CUDA_SAFE(cudaFree(arrg));}
      dims=x.dims; 
      strides=x.strides;
      dev=x.dev;
      memsize=x.memsize; 
      is_view=x.is_view;
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      return *this;
    }
        
    
  public: // ---- Transport -----------------------------------------------------------------------------------


    IntTensor& move_to_device(const int _dev){
      if(dev==_dev) return *this;
      if(is_view) throw std::runtime_error("Cnine error in "+string(__PRETTY_FUNCTION__)+": a tensor view cannot be moved to a different device.");
      memsize=strides[0]*dims[0]; 

      if(_dev==0){
	if(dev==0) return *this;
	assert(arrg);
 	if(arr) delete[] arr;
	arr=new int[std::max(memsize,1)];
	CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(int),cudaMemcpyDeviceToHost));  
	CUDA_SAFE(cudaFree(arrg));
	dev=0;
	return *this;
      }

      if(_dev>0){
	if(dev==_dev) return *this;
	assert(arr);
	if(arrg) CUDA_SAFE(cudaFree(arrg));
	cout<<1321232<<endl;
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
	CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(int),cudaMemcpyHostToDevice));  
	dev=_dev;
	return *this;
      }
      
      return *this;
    }
    
    IntTensor to_device(const int _dev) const{
      return IntTensor(*this,_dev);
    }

    void make_garr(const int _dev=1){
      if(arrg) return;
      assert(arr);
      assert(!is_view);
      cout<<1321232<<endl;
      CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
      CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(int),cudaMemcpyHostToDevice));  
    }


  public: // ---- Dynamic resizing --------------------------------------------------------------------------


    int capacity0() const{
      return memsize/strides[0];
    }

    void reserve0(const int n){
      if(n<=capacity0()) return;
      assert(!is_view);
      memsize=n*strides[0];
      if(memsize==0) memsize=1;
      int asize=strides[0]*dims[0];
      if(dev==0){
	int* newarr=new int[std::max(memsize,1)];
	std::copy(arr,arr+asize,newarr);
	if(arr) delete[] arr;
	arr=newarr;
      }
      if(dev==1){
	int* newarrg=nullptr;
	cout<<1321232<<endl;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, std::max(memsize,1)*sizeof(int)));
	CUDA_SAFE(cudaMemcpy(newarrg,arrg,asize*sizeof(int),cudaMemcpyDeviceToDevice));  
	if(arrg) CUDA_SAFE(cudaFree(arrg));
	arrg=newarrg;
      }
    }

    void resize0(const int n){
      CNINE_ASSRT(!is_view);
      CNINE_ASSRT(n>=dims[0]);
      if(n>capacity0())
	reserve0(n);
      dims[0]=n;
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
    template<typename TYPE>
    IntTensor(const Gtensor<TYPE>& x, const int _dev=0): 
      IntTensor(x.dims,fill::raw){
      assert(x.dev==0);
      int asize=strides[0]*dims[0]; 
      for(int i=0; i<asize; i++){
	arr[i]=x.arr[i];
      }
      move_to_device(_dev);
    }
    
    Gtensor<int> gtensor() const{
      if(dev>0) return IntTensor(*this,0).gtensor();
      Gtensor<int> R(dims,fill::raw);
      int asize=strides[0]*dims[0]; 
      for(int i=0; i<asize; i++){
	R.arr[i]=arr[i];
      }
      return R;
    }

    static IntTensor cat(const vector<reference_wrapper<IntTensor> >& list){
      CNINE_ASSRT(list.size()>0);
      int _dev=0;
      if(list.size()>0) _dev=list[0].get().dev;
      //int subsize=0;
      //if(list.size()>0) subsize=list[0].get().dims.asize()/list[0].get().dims[0]; // TODO
      //for(auto& p:list)
      //CNINE_ASSRT(p.get().dims.asize()/p.get().dims[0]==subsize);

      int t=0;
      for(auto& p:list) t+=p.get().dims[0];
      Gdims D=list[0].get().dims;
      D[0]=t;
      IntTensor R(D);

      int offs=0;
      for(auto& _p:list){
	auto& p=_p.get();
	int asize=p.dims.asize();
	if(_dev==0) std::copy(p.arr,p.arr+asize,R.arr+offs);
	if(_dev==1) CUDA_SAFE(cudaMemcpy(R.arrg+offs,p.arrg,asize*sizeof(int),cudaMemcpyDeviceToDevice));  
	offs+=asize;
      }
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


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

    int* get_arrg(const int _dev=1) const{
      if(!arrg) const_cast<IntTensor&>(*this).make_garr(_dev);
      return arrg;
    }

    int* garr(const int _dev=1) const{
      if(!arrg) const_cast<IntTensor&>(*this).make_garr(_dev);
      return arrg;
    }


  public: // ---- Setting ------------------------------------------------------------------------------------


    void set(const IntTensor& x){
      CNINE_DEVICE_SAME(x);
      CNINE_DIMS_SAME(x);
      int memsize=strides[0]*dims[0]; 
	CPUCODE(std::copy(x.arr,x.arr+memsize,arr));
	GPUCODE(CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(int),cudaMemcpyDeviceToDevice)));  
    }


    void set_zero(){
      int memsize=strides[0]*dims[0]; 
      CPUCODE(std::fill(arr,arr+memsize,0));
      GPUCODE(CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(int))));
    }


  public: // ---- Element Access ------------------------------------------------------------------------------
    

    int operator()(const Gindex& ix) const{
      int t=ix(strides);  
      return arr[t];
    }

    int operator()(const int i0) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      return arr[t];
    }

    int operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      return arr[t];
    }

    int operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return arr[t];
    }

    int operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      return arr[t];
    }

    int operator()(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3] || i4<0 || i4>=dims[4]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3,i4).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4];  
      return arr[t];
    }

    int get(const Gindex& ix) const{
      int t=ix(strides);  
      return arr[t];
    }

    int get(const int i0) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      return arr[t];
    }

    int get(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      return arr[t];
    }

    int get(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      return arr[t];
    }

    int get(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3];  
      return arr[t];
    }

    int get(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      CNINE_CHECK_RANGE(if(dims.size()!=4 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2] || i3<0 || i3>=dims[3] || i4<0 || i4>=dims[4]) 
	  throw std::out_of_range("index "+Gindex(i0,i1,i2,i3,i4).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]+i4*strides[4];  
      return arr[t];
    }


    void set(const Gindex& ix, int x) const{
      int t=ix(strides);  
      arr[t]=x;
    }

    void set(const int i0, int x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=1 || i0<0 || i0>=dims[0]) throw std::out_of_range("index "+Gindex(i0).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0];  
      arr[t]=x;
    }

    void set(const int i0, const int i1, int x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1]) throw std::out_of_range("index "+Gindex(i0,i1).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1];  
      arr[t]=x;
    }

    void set(const int i0, const int i1, const int i2, int x) const{
      CNINE_CHECK_RANGE(if(dims.size()!=3 || i0<0 || i0>=dims[0] || i1<0 || i1>=dims[1] || i2<0 || i2>=dims[2]) throw std::out_of_range("index "+Gindex(i0,i1,i2).str()+" out of range of dimensions "+dims.str()));
      int t=i0*strides[0]+i1*strides[1]+i2*strides[2];  
      arr[t]=x;
    }

    bool operator==(const IntTensor& y) const{
      if(dims!=y.dims) return false;
      if(memsize!=y.memsize) return false;
      for(int i=0; i<memsize; i++)
	if(arr[i]!=y.arr[i]) return false;
      return true;
    }


  public: // ---- Rows ---------------------------------------------------------------------------------------


    vector<int> row(const int i) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i<0 || i>=dims[0]) throw std::out_of_range("row index "+to_string(i)+" out of range of dimensions "+dims.str()));
      vector<int> R(dims[1]);
      for(int j=0; j<dims[1]; j++)
	R[j]=(*this)(i,j);
      return R;
    }

    vector<int> row(const int i, int beg) const{
      CNINE_CHECK_RANGE(if(dims.size()!=2 || i<0 || i>=dims[0]) throw std::out_of_range("row index "+to_string(i)+" out of range of dimensions "+dims.str()));
      vector<int> R(dims[1]-beg);
      for(int j=0; j<dims[1]-beg; j++)
	R[j]=(*this)(i,j+beg);
      return R;
    }

    void set_row(const int i, const initializer_list<int>& list){
      CNINE_ASSRT(dims.size()==2);
      int j=0;
      for(auto p:list)
	set(i,j++,p);
    }

    void push_back(const vector<int>& x){
      CNINE_ASSRT(dims.size()==2);
      CNINE_ASSRT(dims[1]==x.size());
      int row=dims[0];
      //if(capacity0()==0) reserve(1);
      if(capacity0()<=row+1) reserve0(2*(row+1));
      dims[0]++;
      for(int i=0; i<dims[1]; i++)
	set(row,i,x[i]);
    }

    void push_back(const int c, const vector<int>& x){
      CNINE_ASSRT(dims.size()==2);
      CNINE_ASSRT(dims[1]==x.size()+1);
      int row=dims[0];
      if(capacity0()<=row+1) reserve0(2*(row+1));
      dims[0]++;
      set(row,0,c);
      for(int i=0; i<x.size(); i++)
	set(row,i+1,x[i]);
    }

    void push_back(const int c0, const int c1){
      CNINE_ASSRT(dims.size()==2);
      CNINE_ASSRT(dims[1]==2);
      int row=dims[0];
      if(capacity0()<=row+1) reserve0(2*(row+1));
      dims[0]++;
      set(row,0,c0);
      set(row,1,c1);
    }

    void push_back_slice0(const IntTensor& x){
      assert(!is_view);
      assert(dims.size()==x.dims.size()+1);
      for(int i=0; i<x.dims.size(); i++)
	assert(dims[i+1]==x.dims[i]);
      //reserve0(dims[0]+1);
      //if(capacity0()==0) reserve0(1);
      if(capacity0()<dims[0]+1) reserve0(2*(dims[0]+1));
      dims[0]++;
      if(dev==0){
	assert(x.dev==0);
	std::copy(x.arr,x.arr+strides[0],arr+(dims[0]-1)*strides[0]);
      }
      if(dev==1){
	assert(x.dev==1);
	CUDA_SAFE(cudaMemcpy(arrg+(dims[0]-1)*strides[0]+sizeof(int),x.arrg,strides[0]*sizeof(int),cudaMemcpyDeviceToDevice));  
      }
    }


  public: // ---- 1D views -----------------------------------------------------------------------------------


    IntTensor(const Itensor1_view& x):
      IntTensor({x.n0},fill_zero(),x.dev){
      view1().add(x);
    }

    const Itensor1_view view1() const{
      return Itensor1_view(arr,dims,strides);
    }


  public: // ---- 2D views -----------------------------------------------------------------------------------


    IntTensor(const Itensor2_view& x):
      IntTensor({x.n0,x.n1},fill_zero(),x.dev){
      view2().add(x);
    }

    Itensor2_view view2(){
      return Itensor2_view(arr,dims,strides);
    }

    const Itensor2_view view2() const{
      return Itensor2_view(arr,dims,strides);
    }

    Itensor2_view view2(const GindexSet& a, const GindexSet& b){
      return Itensor2_view(arr,dims,strides,a,b);
    }


  public: // ---- 3D views -----------------------------------------------------------------------------------


    IntTensor(const Itensor3_view& x):
      IntTensor({x.n0,x.n1,x.n2},fill_zero(),x.dev){
      view3().add(x);
    }

    Itensor3_view view3(){
      return Itensor3_view(arr,dims,strides);
    }

    const Itensor3_view view3() const{
      return Itensor3_view(arr,dims,strides);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      return gtensor().str(indent);
    }

    string repr() const{
      return "<cnine::IntTensor"+dims.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const IntTensor& x){
      stream<<x.str(); return stream;}
   






  };

}


#endif 
