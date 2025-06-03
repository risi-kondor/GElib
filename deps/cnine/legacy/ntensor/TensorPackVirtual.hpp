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


#ifndef _CnineTensorPackVirtual
#define _CnineTensorPackVirtual

#include "Cnine_base.hpp"
//#include "TensorPackView.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename TYPE, typename BASE>
  class TensorPackVirtual: public BASE{
  public:

    using BASE::arr;
    using BASE::dir;
    using BASE::dev;
    using BASE::dims;
    using BASE::strides;

    using BASE::is_contiguous;


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorPackVirtual(){};
    
    TensorPackVirtual(const Gdims& _dims, const int n, const int _dev=0):
      TensorPackVirtual(TensorPackDir(_dims,n),_dev){}

    TensorPackVirtual(const TensorPackDir& _dir, const int _dev=0): 
      BASE(_dir,MemArr<TYPE>(_dir.total(),_dev)){}

    TensorPackVirtual(const TensorPackDir& _dir, const fill_zero& dummy, const int _dev=0): 
      BASE(_dir,MemArr<TYPE>(_dir.total(),dummy,_dev)){}
    
    TensorPackVirtual(const TensorPackDir& _dir, const fill_sequential& dummy, const int _dev=0):
      TensorPackVirtual(_dir,_dev){
      int N=_dir.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    TensorPackVirtual(const TensorPackDir& _dir, const fill_gaussian& dummy, const int _dev=0):
      TensorPackVirtual(_dir,_dev){
      int N=_dir.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++)
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }

    TensorPackVirtual(const initializer_list<TensorView<TYPE> >& list):
      TensorPackVirtual(TensorPackDir(list)){
      int i=0;
      for(auto& p: list)
	(*this)(i++)=p;
    }

    TensorPackVirtual(const initializer_list<TensorView<TYPE> >& list, const int dev):
      TensorPackVirtual(TensorPackDir(list),dev){
      int i=0;
      for(auto& p: list)
	(*this)(i++)=p;
    }

    TensorPackVirtual(const vector<TensorView<TYPE> >& list):
      TensorPackVirtual(TensorPackDir(list)){
      int i=0;
      for(auto& p: list)
	(*this)(i++)=p;
    }

    TensorPackVirtual(const vector<TensorView<TYPE> >& list, const int dev):
      TensorPackVirtual(TensorPackDir(list),dev){
      int i=0;
      for(auto& p: list)
	(*this)(i++)=p;
    }


    TensorPackVirtual(const vector<Gdims>& _dims, const int _dev=0):
      TensorPackVirtual(TensorPackDir(_dims),_dev){}



  public: // ---- Named constructors ------------------------------------------------------------------------


    static TensorPackVirtual zero(const Gdims& _dims, const int n, const int _dev=0){
      return TensorPackVirtual (TensorPackDir(_dims,n),fill_zero(),_dev);
    }

    static TensorPackVirtual sequential(const Gdims& _dims, const int n, const int _dev=0){
      return TensorPackVirtual(TensorPackDir(_dims,n),fill_sequential(),_dev);
    }

    static TensorPackVirtual gaussian(const Gdims& _dims, const int n, const int _dev=0){
      return TensorPackVirtual(TensorPackDir(_dims,n),fill_gaussian(),_dev);
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    TensorPackVirtual(const vector<const at::Tensor& T>& v):
      TensorPackVirtual(TensorPackDir(v),T.type().is_cuda()){
      (*this)=v;
    }

    #endif 


  public: // ---- Copying ------------------------------------------------------------------------------------


    TensorPackVirtual(const TensorPackVirtual& x):
      TensorPackVirtual(x.dir,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }

    TensorPackVirtual(const TensorPackVirtual& x, const nowarn_flag& dummy):
      TensorPackVirtual(x.dir,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }

    TensorPackVirtual(const TensorPackVirtual&& x):
      BASE(x.dir,x.arr){
      CNINE_MOVE_WARNING();
    }
        
    TensorPackVirtual& operator=(const TensorPackVirtual& x)=delete; 


  public: // ---- Transport ----------------------------------------------------------------------------------


    TensorPackVirtual(const TensorPackVirtual& x, const int _dev):
      TensorPackVirtual(x.dir,_dev){
      view()=x;
    }

    void move_to_device(const int _dev){
      if(dev==_dev) return;
      TensorPackVirtual R(*this,_dev);
      arr=R.arr;
      dev=_dev;
    }


  public: // ---- Views --------------------------------------------------------------------------------------


    TensorPackVirtual(const BASE& x):
      TensorPackVirtual(x.dir,x.dev){
      CNINE_CONVERT_WARNING();
      view()=x;
    }

    BASE view(){
      return BASE(*this);
    }

    const BASE view() const{
      return BASE(*this);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "TensorPackVirtual";
    }


  };

};

#endif 
