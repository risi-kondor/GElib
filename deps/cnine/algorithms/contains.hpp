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


#ifndef _contains
#define _contains

#include "Cnine_base.hpp"


namespace cnine{

  // std::for_any does the same 

  template<typename VEC, typename OBJ>
  bool contains(const VEC& vec, std::function<bool(const OBJ& x)>& lambda){
    for(auto& p:vec)
      if(lambda(p)) return true;
    return false;
  }

}

#endif 
