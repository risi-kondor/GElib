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


#ifndef _CnineTensorPack
#define _CnineTensorPack

#include "Cnine_base.hpp"
#include "TensorPackView.hpp"

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
  class TensorPack: public TensorPackView<TYPE>{
  public:

    typedef TensorPackView<float> TPview;

    using TPview::arr;
    using TPview::dir;
    using TPview::dev;

    using TPview::is_contiguous;

    using TPview::operator();
    //using TPview<TYPE>::set_contiguous;



  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorPack(){};
    
    TensorPack(const int n, const Gdims& _dims, const int _dev=0):
      TensorPack(TensorPackDir(n,_dims),_dev){}

    TensorPack(const Gdims& _dims, const int n, const int _dev=0):
      TensorPack(TensorPackDir(n,_dims),_dev){}

    TensorPack(const TensorPackDir& _dir, const int _dev=0): 
      TPview(_dir,MemArr<TYPE>(_dir.total(),_dev)){
      //set_contiguous(true);
    }

    TensorPack(const TensorPackDir& _dir, const fill_zero& dummy, const int _dev=0): 
      TPview(_dir,MemArr<TYPE>(_dir.total(),dummy,_dev)){
      //set_contiguous(true);
    }
    
    TensorPack(const TensorPackDir& _dir, const fill_sequential& dummy, const int _dev=0):
      TensorPack(_dir,_dev){
      int N=_dir.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    TensorPack(const TensorPackDir& _dir, const fill_gaussian& dummy, const int _dev=0):
      TensorPack(_dir,_dev){
      int N=_dir.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++)
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }

    TensorPack(const initializer_list<TensorView<TYPE> >& list):
      TensorPack(TensorPackDir(list)){
      int i=0;
      for(auto& p: list)
	(*this)(i++)=p;
    }

    TensorPack(const initializer_list<TensorView<TYPE> >& list, const int dev):
      TensorPack(TensorPackDir(list),dev){
      int i=0;
      for(auto& p: list)
	(*this)(i++)=p;
    }

    TensorPack(const vector<TensorView<TYPE> >& list):
      TensorPack(TensorPackDir(list)){
      int i=0;
      for(auto& p: list)
	(*this)(i++)=p;
    }

    TensorPack(const vector<TensorView<TYPE> >& list, const int dev):
      TensorPack(TensorPackDir(list),dev){
      int i=0;
      for(auto& p: list)
	(*this)(i++)=p;
    }


    TensorPack(const vector<Gdims>& _dims, const int _dev=0):
      TensorPack(TensorPackDir(_dims),_dev){}



  public: // ---- Named constructors ------------------------------------------------------------------------


    static TensorPack<TYPE> zero(const int n, const Gdims& _dims, const int _dev=0){
      return TensorPack<TYPE>(TensorPackDir(n,_dims),fill_zero(),_dev);
    }

    static TensorPack<TYPE> sequential(const int n, const Gdims& _dims, const int _dev=0){
      return TensorPack<TYPE>(TensorPackDir(n,_dims),fill_sequential(),_dev);
    }

    static TensorPack<TYPE> gaussian(const int n, const Gdims& _dims, const int _dev=0){
      return TensorPack<TYPE>(TensorPackDir(n,_dims),fill_gaussian(),_dev);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    TensorPack(const TensorPack& x):
      TensorPack(x.dir,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }

    TensorPack(const TensorPack& x, const nowarn_flag& dummy):
      TensorPack(x.dir,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }

    TensorPack(const TensorPack<TYPE>&& x):
      TPview(x.dir,x.arr){
      CNINE_MOVE_WARNING();
    }
        
    TensorPack& operator=(const TensorPack<TYPE>& x)=delete; 


  public: // ---- Transport ----------------------------------------------------------------------------------


    TensorPack(const TensorPack& x, const int _dev):
      TensorPack(x.dir,_dev){
      view()=x;
    }

    void move_to_device(const int _dev){
      if(dev==_dev) return;
      TensorPack R(*this,_dev);
      arr=R.arr;
      dev=_dev;
    }


  public: // ---- Views --------------------------------------------------------------------------------------


    TensorPack(const TPview& x):
      TensorPack(x.dir,x.dev){
      CNINE_CONVERT_WARNING();
      view()=x;
    }

    TPview view(){
      return TPview(*this);
    }

    const TPview view() const{
      return TPview(*this);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "TensorPack";
    }


  };

};

#endif 


    //static TensorPack<TYPE> zero(const Gdims& _dims, const int n, const int _dev=0){
    //return TensorPack<TYPE>(TensorPackDir(_dims,n),fill_zero(),_dev);
    //}

    //static TensorPack<TYPE> sequential(const Gdims& _dims, const int n, const int _dev=0){
    //return TensorPack<TYPE>(TensorPackDir(_dims,n),fill_sequential(),_dev);
    //}

    // static TensorPack<TYPE> gaussian(const Gdims& _dims, const int n, const int _dev=0){
    //return TensorPack<TYPE>(TensorPackDir(_dims,n),fill_gaussian(),_dev);
    //}


