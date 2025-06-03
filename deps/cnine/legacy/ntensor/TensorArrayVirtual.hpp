/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineTensorArrayVirtual
#define _CnineTensorArrayVirtual

#include "Cnine_base.hpp"
#include "TensorArrayView.hpp"

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
  class TensorArrayVirtual: public BASE{
  public:

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::ak;
    using BASE::strides;
    using BASE::dev;
    using BASE::ndims;


  public: // ---- Constructors ------------------------------------------------------------------------------


    //TensorArrayVirtual(const Gdims& _adims, const Gdims& _ddims, const int _dev=0):
    //TensorArrayView<TYPE>(_adims,_ddims,_dev){}


    // DO NOT USE THESE

    /*
    TensorArrayVirtual(){};

    TensorArrayVirtual(const int _ak, const Gdims& _dims, const int _dev=0){
      arr=MemArr<TYPE>(_dims.total(),_dev);
      dims=_dims; 
      strides=GstridesB(_dims); 
      dev=_dev;
      ak=_ak;
    }
    //arr(MemArr<TYPE>(_dims.total(),_dev)),
    //dims(_dims), 
    //strides(GstridesB(_dims)), 
    //dev(_dev),
    //ak(_ak){}

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _ddims, const int _dev=0):
      TensorArrayVirtual(_adims.size(),_adims.cat(_ddims),_dev){}

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _ddims, const fill_zero& dummy, const int _dev=0):
      TensorArrayVirtual(_adims,_ddims,_dev){
      // TODO
    }

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _ddims, const fill_constant<TYPE>& dummy, const int _dev=0):
      TensorArrayVirtual(_adims,_ddims,_dev){
      // TODO
    }

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _ddims, const fill_sequential& dummy, const int _dev=0):
      TensorArrayVirtual(_adims,_ddims,_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _ddims, const fill_gaussian& dummy, const int _dev=0):
      TensorArrayVirtual(_adims,_ddims,_dev){
      int N=dims.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }
    */

    //template<typename FILLTYPE, typename = typename 
    //     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //TensorArrayVirtual(const Gdims& _adims, const Gdims& _ddims, const FILLTYPE& fill, const int _dev=0):
    //BASE(_adims,_ddims,fill,_dev){cout<<8878<<endl;}


  public: // ---- Named constructors ------------------------------------------------------------------------


  public: // ---- Copying -----------------------------------------------------------------------------------


    // need a move too!
    TensorArrayVirtual(const TensorArrayVirtual& x):
      TensorArrayVirtual(x.ak,x.dims,x.dev){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    TensorArrayVirtual(const TensorArrayVirtual& x, const nowarn_flag& dummy):
      TensorArrayVirtual(x.dims,x.dev){
      view()=x.view();
    }
        
    TensorArrayVirtual(const TensorArrayVirtual&& x):
      BASE(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
    }
        
    TensorArrayVirtual& operator=(const TensorArrayVirtual& x){
      arr=x.arr;
      return *this;
    }

    
  public: // ---- Conversions ---------------------------------------------------------------------------------

    /* caused construction problems
    TensorArrayVirtual(const BASE& x):
      TensorArrayVirtual(x.get_adims(),x.get_ddims(),x.dev){
      CNINE_CONVERT_WARNING();
      view()=x;
    }
    */

  public: // ---- Transport -----------------------------------------------------------------------------------


    TensorArrayVirtual(const BASE& x, const int _dev):
      TensorArrayVirtual(x.dims,_dev){
      CNINE_COPY_WARNING();
      view()=x;
    }

    void move_to_device(const int _dev) const{
      if(dev==_dev) return;
      const_cast<TensorArrayVirtual&>(*this)=TensorArrayVirtual(*this,_dev);
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    //TensorArrayVirtual(const at::Tensor& T):
    //TensorArrayVirtual(Gdims(T),T.type().is_cuda()){
    //BASE::operator=(T);
    //}

    #endif


  public: // ---- Views -------------------------------------------------------------------------------------


    BASE view(){
      return BASE(*this);
    }

    const BASE view() const{
      return BASE(*this);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    /*
    TensorArrayVirtual operator*(const BASE& y) const{
      CNINE_ASSERT(ndims()==1||ndims()==2,"first operand of product must be a vector or a matrix");
      CNINE_ASSERT(y.ndims()==1||y.ndims()==2,"second operand of product must be a vector or a matrix");

      if(ndims()==1 && y.ndims()==2){
	TensorArrayVirtual R=zero({y.dims[1]},dev);
	R.add_mvprod_T(y,*this);
	return R;
      }

      if(ndims()==2 && y.ndims()==1){
	TensorArrayVirtual R=zero({dims[0]},dev);
	R.add_mvprod(*this,y);
	return R;
      }

      if(ndims()==2 && y.ndims()==2){
	TensorArrayVirtual R=zero({dims[0],y.dims[1]},dev);
	R.add_mprod(*this,y);
	return R;
      }

      return TensorArrayVirtual();
    }
    */


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "TensorArrayVirtual";
    }


  };

}
    
#endif 


    // need this?
    //TensorArrayVirtual(const Gdims& _dims, const int _dev=0): 
    //BASE(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

    // need this?
    //TensorArrayVirtual(const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
    //BASE(MemArr<TYPE>(_dims.total(),dummy,_dev),_dims,GstridesB(_dims)){}

    // need this?
    /*
    TensorArrayVirtual(const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      TensorArrayVirtual(_dims,_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }
    */

    // need this?
    /*
    TensorArrayVirtual(const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      TensorArrayVirtual(_dims,_dev){
      int N=dims.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }
    */

    /*
    TensorArrayVirtual(const Gdims& _adims, const Gdims& _dims, const int _dev=0): 
      BASE(MemArr<TYPE>(_dims.total(),_dev),_dims,GstridesB(_dims)){}

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _dims, const fill_zero& dummy, const int _dev=0): 
      BASE(MemArr<TYPE>((_adims.cat(_dims)).total(),dummy,_dev),_adims.cat(_dims),GstridesB(_adims.cat(_dims))){}

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _dims, const fill_sequential& dummy, const int _dev=0):
      TensorArrayVirtual(_adims.cat(_dims),_dev){
      int N=dims.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    TensorArrayVirtual(const Gdims& _adims, const Gdims& _dims, const fill_gaussian& dummy, const int _dev=0):
      TensorArrayVirtual(_adims.cat(_dims),_dev){
      int N=dims.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++) 
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }
    */
    //static TensorArrayVirtual zero(const Gdims& _dims, const int _dev=0){
    //return TensorArrayVirtual (_dims,fill_zero(),_dev);
    //}

    //static TensorArrayVirtual sequential(const Gdims& _dims, const int _dev=0){
    //return TensorArrayVirtual(_dims,fill_sequential(),_dev);
    //}

    //static TensorArrayVirtual gaussian(const Gdims& _dims, const int _dev=0){
    //return TensorArrayVirtual(_dims,fill_gaussian(),_dev);
    //}


