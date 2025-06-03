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

#ifndef _DeltaFactor
#define _DeltaFactor

#include "Cnine_base.hpp"
#include "../containers/object_bank.hpp"
#include "../math/frational.hpp"
#include "../math/FFactorial.hpp"

namespace cnine{

  class DeltaSignature{
  public:
    int a,b,c;
    DeltaSignature(const int _a, const int _b, const int _c): a(_a), b(_b), c(_c){}
    bool operator==(const DeltaSignature& x) const{
      return (a==x.a)&&(b==x.b)&&(c==x.c);
    }
  };

}

namespace std{
  template<>
  struct hash<cnine::DeltaSignature>{
  public:
    size_t operator()(const cnine::DeltaSignature& x) const{
      size_t h=hash<int>()(x.a);
      h=(h<<1)^hash<int>()(x.b);
      h=(h<<1)^hash<int>()(x.c);
      return h;
    }
  };
}


namespace cnine{

  extern FFactorial ffactorial;

  class DeltaFactor: public object_bank<DeltaSignature,frational>{
  public:

    using  object_bank<DeltaSignature,frational>::object_bank; //<DeltaSignature,frational>;
    using  object_bank<DeltaSignature,frational>::operator();

    DeltaFactor():
      object_bank<DeltaSignature,frational>([](const DeltaSignature& x){
	  const int a=x.a;
	  const int b=x.b;
	  const int c=x.c;

	  //cout<<"Delta("<<a<<","<<b<<","<<c<<")"<<endl;
	  frational R=ffactorial(a+b-c);
	  return new frational(R*ffactorial(a-b+c)*ffactorial(-a+b+c)/ffactorial(a+b+c+1));
	}){}

    double operator()(const int a, const int b, const int c){
      return sqrt((*this)(DeltaSignature(a,b,c)));
    }

    frational squared(const int a, const int b, const int c){
      return (*this)(DeltaSignature(a,b,c));
    }

  };

}

#endif 
