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


#ifndef __Lgrid
#define __Lgrid

#include "Cnine_base.hpp"
#include "Ldims.hpp"

namespace cnine{


  class Lgrid: public Ldims{
  public:

    Lgrid(const vector<int>& x):
      Ldims(x){
    }

  public: // ---- Copying ------------------------------------------------------------------------------------


    virtual Lgrid* clone() const{
      return new Lgrid(*this);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    virtual string name() const{
      return "grid";
    }

    virtual string str() const{
      return Ldims::str();
    }

    friend ostream& operator<<(ostream& stream, const Lgrid& x){
      stream<<x.str(); return stream;}

  };


  // dangerous hack
  inline Lgrid* grid(const vector<int>& x){
    return new Lgrid(x);
  }

}

#endif 

