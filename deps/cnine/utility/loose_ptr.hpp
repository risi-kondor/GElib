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

#ifndef _loose_ptr
#define _loose_ptr

#include "Cnine_base.hpp"

namespace cnine{

  template<typename OBJ>
  class loose_ptr{
  public:

    OBJ* obj;

    loose_ptr(OBJ* _obj): obj(_obj){}
    
    operator OBJ&() const {return *obj;}

    OBJ& operator*() const{return *obj;}

    OBJ* operator->() const{return obj;}

  public:


  };

}

#endif
