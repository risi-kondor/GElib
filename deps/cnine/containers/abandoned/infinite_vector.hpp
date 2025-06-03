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

#ifndef _infinite_vector
#ifndef _infinite_vector

#include "Cnine_base.hpp"


namespace cnine{

  template<typename TYPE>
  class infinite_vector: vector<TYPE>{
  public:

    typename vector<TYPE> BASE;

    using BASE::BASE;
    using BASE::size;
    using BASE::resize;
    using BASE::operator[];


  public:

    TYPE operator[](const int i) const{
      if(i>=size()) resize(i+1);
      return (*this)[i];
    }
    
    TYPE& operator[](const int i){
      if(i>=size()) resize(i+1);
      return (*this)[i];
    }
    
  };

}

#endif 
