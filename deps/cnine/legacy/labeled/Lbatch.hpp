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


#ifndef __Lbatch
#define __Lbatch

#include "Cnine_base.hpp"
#include "Ldims.hpp"

namespace cnine{


  class Lbatch: public Ldims{
  public:

    Lbatch(const int x):
      Ldims({x}){
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    virtual Lbatch* clone() const{
      return new Lbatch(*this);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    virtual string name() const{
      return "batch";
    }

    virtual string str() const{
      return "batch("+to_string((*this)[0])+")";
    }

    friend ostream& operator<<(ostream& stream, const Lbatch& x){
      stream<<x.str(); return stream;}

  };


  // dangerous hack
  inline Lbatch* batch(const int B){
    return new Lbatch(B);
  }

}

#endif 

