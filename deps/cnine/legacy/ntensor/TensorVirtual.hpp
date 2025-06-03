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


#ifndef _CnineTensorVirtual
#define _CnineTensorVirtual

#include "Cnine_base.hpp"
#include "TensorView.hpp"

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
  class TensorVirtual: public BASE{
  public:

    typedef std::size_t size_t;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;
    using BASE::ndims;


  public: // ---- Constructors ------------------------------------------------------------------------------


    TensorVirtual(){};

    TensorVirtual(const Gdims& _dims, const int _dev=0): 
      BASE(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

    TensorVirtual(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      BASE(MemArr<TYPE>(_dims.total(),dummy,_dev),_dims,GstridesB(_dims)){}

    TensorVirtual(const Gdims& _dims, const fill_constant<TYPE>& dummy, const int _dev=0): 
      BASE(MemArr<TYPE>(_dims.total(),dummy,_dev),_dims,GstridesB(_dims)){}

    TensorVirtual(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      TensorVirtual(_dims,_dev){
      size_t N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    TensorVirtual(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      TensorVirtual(_dims,_dev){
      size_t N=dims.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }


  public: // ---- Named constructors ------------------------------------------------------------------------


    static TensorVirtual zero(const Gdims& _dims, const int _dev=0){
      return TensorVirtual(_dims,fill_zero(),_dev);
    }

    static TensorVirtual constant(const Gdims& _dims, const TYPE& v, const int _dev=0){
      return TensorVirtual(_dims,fill_constant<TYPE>(v),_dev);
    }

    static TensorVirtual sequential(const Gdims& _dims, const int _dev=0){
      return TensorVirtual(_dims,fill_sequential(),_dev);
    }

    static TensorVirtual gaussian(const Gdims& _dims, const int _dev=0){
      return TensorVirtual(_dims,fill_gaussian(),_dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    TensorVirtual(const TensorVirtual& x):
      TensorVirtual(x.dims,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    TensorVirtual(const TensorVirtual& x, const nowarn_flag& dummy):
      TensorVirtual(x.dims,x.dev){
      view()=x.view();
    }
        
    TensorVirtual(const TensorVirtual&& x):
      BASE(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
    }
        
    TensorVirtual& operator=(const TensorVirtual& x){
      arr=x.arr;
      return *this;
    }
    

  public: // ---- Transport -----------------------------------------------------------------------------------


    TensorVirtual(const BASE& x, const int _dev):
      TensorVirtual(x.dims,_dev){
      CNINE_COPY_WARNING();
      view()=x;
    }

    void move_to_device(const int _dev) const{
      if(dev==_dev) return;
      const_cast<TensorVirtual&>(*this)=TensorVirtual(*this,_dev);
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    //TensorVirtual(const at::Tensor& T):
    //TensorVirtual(Gdims(T),T.type().is_cuda()){
    //cout<<77<<endl;
    //BASE::
    //(*this)=T;
    //cout<<88<<endl;
    //}

    #endif


  public: // ---- Views -------------------------------------------------------------------------------------


    TensorVirtual(const BASE& x):
      TensorVirtual(x.dims,x.dev){
      CNINE_CONVERT_WARNING();
      view()=x;
    }

    BASE view(){
      return BASE(*this);
    }

    const BASE view() const{
      return BASE(*this);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    TensorVirtual operator*(const BASE& y) const{
      CNINE_ASSERT(ndims()==1||ndims()==2,"first operand of product must be a vector or a matrix");
      CNINE_ASSERT(y.ndims()==1||y.ndims()==2,"second operand of product must be a vector or a matrix");

      if(ndims()==1 && y.ndims()==2){
	TensorVirtual R=zero({y.dims[1]},dev);
	R.add_mvprod_T(y,*this);
	return R;
      }

      if(ndims()==2 && y.ndims()==1){
	TensorVirtual R=zero({dims[0]},dev);
	R.add_mvprod(*this,y);
	return R;
      }

      if(ndims()==2 && y.ndims()==2){
	TensorVirtual R=zero({dims[0],y.dims[1]},dev);
	R.add_mprod(*this,y);
	return R;
      }

      return TensorVirtual();
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorVirtual";
    }

    //string describe() const{
    //ostringstream oss;
    //oss<<"TensorVirtual"<<dims<<" ["<<strides<<"]"<<endl;
    //return oss.str();
    //}

    

    //friend ostream& operator<<(ostream& stream, const TensorVirtual& x){
    //stream<<x.str(); return stream;
    //}

  };


}

#endif
