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


#ifndef _CnineCtensorA_setIdentity
#define _CnineCtensorA_setIdentity

#include "CtensorA_cop.hpp"


namespace cnine{

  class CtensorA_setIdentity_cop: public CtensorA_inplace_cop{
  public:
    
    void operator()(const CtensorA_spec& spec, float* arr, float* arrc) const{
      std::fill(arr,arr+spec.asize,0);
	std::fill(arrc,arrc+spec.asize,0);
	for(int i=0; i<spec.dims[spec.k-1]; i++)
	  arr[i*(spec.strides[spec.k-2]+1)]=1;
    }
    
    
  };
  
}


#endif 
