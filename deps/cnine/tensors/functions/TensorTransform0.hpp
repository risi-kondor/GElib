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


#ifndef _CnineTensorTransform1
#define _CnineTensorTransform1

#include "TensorView.hpp"
#include "EinsumForm1.hpp"
#include "Einsum1params.hpp"
#include "GatherMapB.hpp"


namespace cnine{
  
  template<typename TYPE>
  class TensorTransform1{
  public:

    TensorTransform1(){}

    void operator()(TYPE* xarr, TYPE* rarr, const int xn0, const int xs0, const int rn0, const int rs0){
    }

  };

}
