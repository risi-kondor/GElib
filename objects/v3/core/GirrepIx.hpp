
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGirrepIx
#define _GElibGirrepIx

#include "GElib_base.hpp"


namespace GElib{


  class GirrepIx{
  public:
    
    virtual ~GirrepIx(){}

    virtual GirrepIx* clone() const=0;

    virtual bool operator==(const GirrepIx& y) const=0;

    virtual bool operator<(const GirrepIx& y) const=0;


  public: // ---- I/O ----------------------------------------------------------------------------------------


    virtual string classname() const{
      return "GElib::GirrepIx";
    }

    virtual string repr() const=0; 
    
    virtual string str() const=0;

    friend ostream& operator<<(ostream& stream, const GirrepIx& x){
      stream<<x.str(); return stream;
    }


  };


  class GirrepIxWrapper{
  public:

    shared_ptr<GirrepIx> p;

    GirrepIxWrapper(GirrepIx* _p):
      p(_p){}

    GirrepIxWrapper(const shared_ptr<GirrepIx>& _p):
      p(_p){}

    bool operator=(const GirrepIxWrapper& y) const{
      return *p==*y.p;
    }

    bool operator<(const GirrepIxWrapper& y) const{
      return *p<*y.p;
    }

    string str() const{
      return p->str();
    }

    friend ostream& operator<<(ostream& stream, const GirrepIxWrapper& x){
      stream<<x.str(); return stream;
    }
    
  };


}

#endif 
