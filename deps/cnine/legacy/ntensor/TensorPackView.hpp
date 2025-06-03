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


#ifndef _CnineTensorPackView
#define _CnineTensorPackView

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "GstridesB.hpp"
#include "Gindex.hpp"
#include "MemArr.hpp"
#include "TensorPackDir.hpp"
#include "TensorView.hpp"
#include "device_helpers.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE>
  class TensorPackView{
  public:

    MemArr<TYPE> arr;
    TensorPackDir dir;
    int dev=0;
    //bool contiguous=false;


  public: // ---- Constructors ------------------------------------------------------------------------------


    ///TensorPackView(){
    //cout<<"init2"<<endl;
    //}

    TensorPackView(const TensorPackDir& _dir, const MemArr<TYPE>& _arr):
      arr(_arr),
      dir(_dir),
      dev(_arr.device()){
    }


  public: // ---- Copying ------------------------------------------------------------------------------------

    
    TensorPackView(const TensorPackView& x):
      arr(x.arr),
      dir(x.dir),
      dev(x.dev){}
    //contiguous(x.contiguous){}

    TensorPackView(TensorPackView&& x):
      arr(x.arr),
      dir(std::move(x.dir)),
      dev(x.dev){}
    //contiguous(x.contiguous){}

    TensorPackView& operator=(const TensorPackView& x){
      CNINE_ASSRT(size()==x.size());
      for(int i=0; i<size(); i++)
	(*this)[i]=x[i];
      return *this;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    // deprecated
    //TensorPackView(const vector<const at::Tensor& T>& v):
    //TensorPackView(TensorPackDir(v),T.type().is_cuda()){
    //}
	
    TensorPackView& operator=(const vector<at::Tensor>& v){
      for(int i=0; i<size(); i++)
	(*this)[i]=v[i];
    }

    #endif 


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return dir.size();
    }

    bool is_contiguous() const{
      return dir.is_contiguous();
    }

    int uniform_last_dim() const{
      return dir.uniform_last_dim();
    }

    //TensorPackView& set_contiguous(const bool x){
    //contiguous=x;
    //return *this;
    //}

    int total() const{
      return dir.total();
    }

    int offset() const{
      if(size()==0) return 0;
      return offset(0);
    }

    TYPE* mem() const{
      if(size()==0) return const_cast<TYPE*>(arr.get_arr());
      return const_cast<TYPE*>(arr.get_arr())+offset(0);
    }


  public: // individual tensors


    int offset(const int i) const{
      return dir.offset(i);
    }

    Gdims dims(const int i) const{
      return dir.dims(i);
    }

    GstridesB strides(const int i) const{
      return dir.strides(i);
    }

    TensorView<TYPE> operator()(const int i) const{
      return TensorView<TYPE>(arr+offset(i),dims(i),strides(i));//.set_offset(0));
    }

    TensorView<TYPE> operator[](const int i) const{
      return TensorView<TYPE>(arr+offset(i),dims(i),strides(i));//.set_offset(0));
    }


  public: // ---- Fusing ------------------------------------------------------------------------------------

    
    TensorView<TYPE> fuse() const{
      CNINE_ASSRT(is_contiguous());
      return TensorView<TYPE>(arr,{total()},GstridesB({1})/*.set_offset(offset())*/);
    }

    TensorView<TYPE> fuse_with_all_but_last() const{
      CNINE_ASSRT(is_contiguous());
      CNINE_ASSRT(uniform_last_dim());
      const int m=uniform_last_dim();
      return TensorView<TYPE>(arr,{total()/m,m},GstridesB({m,1})/*.set_offset(offset())*/);
    }


  public: // ---- Lambdas -----------------------------------------------------------------------------------


    void for_each(const std::function<void(const int i, const TensorView<TYPE>&)>& lambda) const{
      for(int i=0; i<size(); i++)
	lambda(i,(*this)[i]);
    }

    void zip(const TensorPackView& x, const std::function<void(const TensorView<TYPE>&, const TensorView<TYPE>&, const int i)>& lambda) const{
      CNINE_ASSRT(x.size()==size());
      for(int i=0; i<size(); i++)
	lambda((*this)[i],x[i],i);
    }

    void zip(const TensorPackView& x, const TensorPackView& y, 
      const std::function<void(const TensorView<TYPE>&, const TensorView<TYPE>&, const TensorView<TYPE>&, const int i)>& lambda) const{
      CNINE_ASSRT(x.size()==size());
      CNINE_ASSRT(y.size()==size());
      for(int i=0; i<size(); i++)
	lambda((*this)[i],x[i],y[i],i);
    }


  public: // ---- In-place Operations ------------------------------------------------------------------------


    void set_zero() const{
      if(is_contiguous()) fuse().set_zero();
      else for_each([&](const int i, const TensorView<TYPE>& x){x.set_zero();});
    }

    void inplace_times(const TYPE c) const{
      if(is_contiguous()) fuse().inplace_times(c);
      else for_each([&](const int i, const TensorView<TYPE>& x){x.inplace_times(c);});
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add(const TensorPackView& x){
      CNINE_DEVICE_SAME(x);
      //CNINE_CHECK_SIZE(dims.check_dims_equal(x.dims));
      if(is_contiguous() && x.is_contiguous()) fuse().add(x.fuse());
      else zip(x,[&](const TensorView<TYPE>& x, const TensorView<TYPE>& y, const int i){x.add(y);});
    }

    void add(const TensorPackView& x, const TYPE c){
      CNINE_DEVICE_SAME(x);
      //CNINE_CHECK_SIZE(dims.check_dims_equal(x.dims));
      if(is_contiguous() && x.is_contiguous()) fuse().add(x.fuse(),c);
      else zip(x,[&](const TensorView<TYPE>& x, const TensorView<TYPE>& y, const int i){x.add(y,c);});
    }


  public: // ---- Matrix multiplication ---------------------------------------------------------------------


    void add_mvprod(const TensorPackView& x, const TensorPackView& y) const{
      CNINE_DEVICE_SAME(x);
      CNINE_DEVICE_SAME(y);
      zip(x,y,[&](const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y, const int i){
	  r.add_mvprod(x,y);});
    }

    void add_mvprod_T(const TensorPackView& x, const TensorPackView& y) const{
      CNINE_DEVICE_SAME(x);
      CNINE_DEVICE_SAME(y);
      zip(x,y,[&](const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y, const int i){
	  r.add_mvprod_T(x,y);});
    }

    void add_mprod(const TensorPackView& x, const TensorPackView& y) const{
      CNINE_DEVICE_SAME(x);
      CNINE_DEVICE_SAME(y);
      zip(x,y,[&](const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y, const int i){
	  r.add_mprod(x,y);});
    }


    void add_mvprod(const TensorPackView& x, const TensorView<TYPE>& y) const{
      CNINE_DEVICE_SAME(x);
      CNINE_DEVICE_SAME(y);
      CNINE_ASSRT(x.size()==size());

      zip(x,y,[&](const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y, const int i){
	  r.add_mvprod(x,y);});
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "TensorPackView";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Tensor "<<i<<":"<<endl;
	oss<<(*this)[i].str(indent+"  ")<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const TensorPackView& v){
      stream<<v.str(); return stream;}


  };

}

#endif
      //if(dev==0){
      //  std::fill(mem(),mem()+total(),0);
      //}
      //if(dev==1){
      // CUDA_SAFE(cudaMemset(mem(),0,total()*sizeof(TYPE)));
      //}
      //}
      //if(dev==0){
      //for(int i=0; i<total(); i++)
      //*(mem()+i)*=c;
      //}
      //if(dev==1){
      //  CNINE_UNIMPL();
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas,total(),&c,mem(), 1,mem(), 1));
      //}
      /*
	if(dev==0){
	stdadd(x.mem(),x.mem()+x.total(),mem());
	}
	if(dev==1){
	CNINE_UNIMPL();
	const TYPE alpha=1.0;
	CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize, &alpha, x.arrg, 1, arrg, 1));
	}
      */
