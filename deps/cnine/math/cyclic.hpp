/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _cyclic
#define _cyclic

#include "Cnine_base.hpp"
#include "Primes.hpp"

namespace cnine{


  template<typename TYPE>
  class cyclic{
  public:

    TYPE v=0;
    TYPE n;

    cyclic(const int _n, const int _v=0): 
      n(_n), v(_v%_n){}


  public: // ------------------------------------------------------------------------------------------------


    static cyclic random(const int _n){
      std::uniform_int_distribution<> distr(0,_n-1);
      return cyclic(_n,distr(rndGen));
    }


  public: // ------------------------------------------------------------------------------------------------


    TYPE get() const{
      return v;
    }

    cyclic operator+(const cyclic& y) const{
      return cyclic(n,v+y.v);
    }

    cyclic operator-(const cyclic& y) const{
      return cyclic(n,v-y.v);
    }

    cyclic operator-() const{
      return cyclic(n,n-v);
    }


  public: // -------------------------------------------------------------------------------------------------


    string str() const{
      return to_string(v);
    }

    friend ostream& operator<<(ostream& stream, const cyclic<TYPE>& x){
      stream<<x.get(); return stream;}

  };


  template<typename TYPE>
  inline string to_string(const cyclic<TYPE>& x){
    return std::to_string(x.v);
  }

}

#endif
