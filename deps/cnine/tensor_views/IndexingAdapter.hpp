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


#ifndef _IndexingAdapeter
#define _IndexingAdapeter

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"
#include "Gindex.hpp"


namespace cnine{



  template<typename OBJ, typename TRANSF0>
  class IndexingAdapter1: public OBJ{
  public:

    const OBJ& obj;
    const TRANSF0& T0;

    IndexingAdapter1(const OBJ& _obj, const TRANSF0& _T0):
      obj(_obj), T0(_T0){}


  public: // ---- Getters -----------------------------------------------------------------------------------


    //float operator()(const int i0){
    //return obj(TO(i0));
    //}

    float operator()(const int i0){
      return obj(T0(i0));
    }


    float operator()(const int i0, const int i1){
      return obj(T0(i0,i1));
    }

    float operator()(const int i0, const int i1, const int i2){
      return obj(T0(i0,i1,i2));
    }


  public: // ---- Conversions --------------------------------------------------------------------------------

    
    template<typename TENSOR> 
    TENSOR to() const{
      TENSOR R=TENSOR::zero(T0.get_dims());
      cout<<R<<endl;
      T0.foreach([&](const Gindex& ix){
	  R.set(ix,obj(T0(ix)));
	});
      return R;
    }


  };

}

#endif
