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


#ifndef _TensorView
#define _TensorView

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"


namespace cnine{


  template<typename TYPE>
  class TensorView{
  public:

    TYPE* arr;
    const Gdims dims;
    const Gstrides strides;
    int dev=0;
    bool regular=false;

  public:

    //TensorView(){}

    //TensorView(float* _arr): 
    //arr(_arr){}

    TensorView(TYPE* _arr,  const Gdims& _dims, const Gstrides& _strides, const int _dev=0):
      arr(_arr), dims(_dims), strides(_strides), dev(_dev){
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    TensorView(const TensorView<TYPE>& x):
      arr(x.arr), dims(x.dims), strides(x.strides), dev(x.dev), regular(strides.is_regular(dims)){}

    TensorView& operator=(const TensorView<TYPE>& x){
      PTENS_ASSRT(dims==x.dims);
      if(is_regular()&& x.is_regular()){
	CNINE_CPUONLY();
	CPUCODE(std::copy(x.arr,x.arr+x.memsize(),arr));
      }else{
	CNINE_CPUONLY();
	x.for_each([&](const Gindex& ix, const TYPE x){set(ix,x);});
      }
    }

    
  public: // ---- Access -------------------------------------------------------------------------------------


    int ndims() const{
      return dims.size();
    }

    bool is_regular() const{
      return regular;
    }

    int asize() const{
      return dims.asize();
    }

    int memsize() const{
      return strides.memsize(dims);
    }


  public: // ---- Getters ------------------------------------------------------------------------------------


    TYPE operator()(const Gindex& ix) const{
      return arr[ix(strides)];
    }

    TYPE operator()(const int i0) const{
      CNINE_DIMS(1);
      CNINE_CHECK_RANGE(if(i0<0 || i0>=dims[0]) 
	  throw std::out_of_range("cnine::TensorView: index "+Gindex({i0}).str()+" out of range of view size "+dims.str()));
      return arr[strides.offs(i0)];
    }

    TYPE operator()(const int i0, const int i1) const{
      CNINE_DIMS(2);
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=dims[0] || i1>=dims[1]) 
	  throw std::out_of_range("cnine::TensorView: index "+Gindex({i0,i1}).str()+" out of range of view size "+dims.str()));
      return arr[strides.offs(i0,i1)];
    }

    TYPE operator()(const int i0, const int i1, const int i2) const{
      CNINE_DIMS(3);
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2]) 
	  throw std::out_of_range("cnine::TensorView: index "+Gindex({i0,i1,i2}).str()+" out of range of view size "+dims.str()));
      return arr[strides.offs(i0,i1,i2)];
    }

    TYPE operator()(const int i0, const int i1, const int i2, const int i3) const{
      CNINE_DIMS(4);
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i2<0 || i3<0 || i0>=dims[0] || i1>=dims[1] || i2>=dims[2] || i3>=dims[3]) 
	  throw std::out_of_range("cnine::TensorView: index "+Gindex({i0,i1,i2,i3}).str()+" out of range of view size "+dims.str()));
      return arr[strides.offs(i0,i1,i2,i3)];
    }


  public: // ---- Setters ------------------------------------------------------------------------------------


    void set(const Gindex& ix, const TYPE x) const{
      arr[ix(strides)]=x;
    }


  public: // ---- Lambdas -------------------------------------------------------------------------------------


    void for_each(const std::function<void(const Gindex&, const TYPE)>& lambda){
      int a=asize();
      if(regular){
	for(int i=0; i<a; i++)
	  lambda(Gindex(i,strides),arr[i]);
      }else{
	CNINE_UNIMPL();
      }
    }

  };

}

#endif
