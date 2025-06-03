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

#ifndef _mtensor_signature
#define _mtensor_signature

#include "Cengine_base.hpp"


namespace cnine{


  //template<typename TYPE>
  class mtensor_signature{
  public:

    Gdims dims;

    mtensor_signature(const Gdims& _dims): 
      dims(_dims){
    }

    bool operator==(const mtensor_signature& x) const{
      return (dims==x.dims);}

    string str() const{
      return "("+dims.str()+")";}

  };
  
}


namespace std{

  template<>
  struct hash<::cnine::mtensor_signature>{
  public:
    size_t operator()(const ::cnine::mtensor_signature& ix) const{
      size_t t=(hash<::cnine::Gdims>()(ix.dims)<<1);
      return t;
    }
  };

}


#endif
