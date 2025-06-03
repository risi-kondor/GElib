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

#ifndef _ATview
#define _ATview

#include "Cnine_base.hpp"
#include "Ltensor.hpp"


namespace cnine{

#ifdef _WITH_ATEN

  // This is really dangerous and should not be used

  template<typename TYPE>
  class ATview: public Ltensor<TYPE>{
  public:
    
    typedef Ltensor<TYPE> BASE;

    ATview(at::Tensor& x):
      BASE(BASE::view(x)){}

    ~ATview(){
      BASE::arr.blob->arr=nullptr;
    }

  };


#endif 

}

#endif 
