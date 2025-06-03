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


#ifndef _NamedType
#define _NamedType

namespace cnine{

  template<typename TYPE, typename TAG>
  class NamedType{
  public:

    explicit NamedType(const TYPE& value): value_(value){}

    explicit NamedType(TYPE&& value): value_(std::move(value)){}

    TYPE& get(){return value_;}

    const TYPE& get() const{return value_;}

    struct argument{
      template<typename BASE>
      NamedType operator=(BASE&& value) const{
	return NamedType(std::forward<BASE>(value));
      }
    };


  private:

    TYPE value_;

  };


}

#endif 
