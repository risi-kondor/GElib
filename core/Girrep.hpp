/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GElibGirrep
#define _GElibGirrep

#include "GElib_base.hpp"
#include "GirrepObj.hpp"
#include "Gelement.hpp"


namespace GElib{


  class Girrep{
  public:

    const shared_ptr<GirrepObj> obj;


    Girrep(GirrepObj* x):
      obj(x){}

    Girrep(const shared_ptr<GirrepObj>& x):
      obj(x){}

    virtual ~Girrep(){}


  public: // ----- Access -----------------------------------------------------------------------------------


    int dim() const{
      return obj->dim();
    }

    bool operator<(const Girrep& y) const{
      return (*obj)<(*y.obj);
    }


  public: // ----- Operations --------------------------------------------------------------------------------


    tensor operator()(const Gelement& x){
      return (*obj)(*x.obj);
    }
    

  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str() const{
      return obj->str();
    }

    string repr() const{
      return obj->str();
    }

    friend ostream& operator<<(ostream& stream, const Girrep& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 

