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

#ifndef _GElibIrrepObj
#define _GElibIrrepObj

#include "GElib_base.hpp"
#include "GelementObj.hpp"
//#include "GirrepIxObj.hpp"
//#include "Ltensor.hpp"
//#include "Dtensor.hpp"


namespace Elib{


  class GirrepObj{
  public:


    virtual ~GirrepObj(){}


  public: // ---- Operations ---------------------------------------------------------------------------------


    virtual int dim() const=0;

    virtual bool operator==(const GirrepObj& y) const=0;

    virtual bool operator<(const GirrepObj& y) const=0;

    // virtual tensor operator()(const GelementObj& x) const=0;


  public: // ---- CG-product --------------------------------------------------------------------------------


    virtual int CGmultiplicity(const GirrepObj& x, const GirrepObj& y){
      GELIB_UNIMPL();
      return 0;
    }

    /*
    virtual void addCGproduct(cnine::Dtensor& r, const cnine::Dtensor& x, const cnine::Dtensor& y, const int offs=0){
      GENET_UNIMPL();
    }

    virtual void addCGproduct_back0(cnine::Dtensor& r, const cnine::Dtensor& g, const cnine::Dtensor& y, const int offs=0){
      GENET_UNIMPL();
    }

    virtual void addCGproduct_back1(cnine::Dtensor& r, const cnine::Dtensor& g, const cnine::Dtensor& x, const int offs=0){
      GENET_UNIMPL();
    }
    */


  public: // ---- I/O --------------------------------------------------------------------------------------


    virtual string repr() const=0;

    virtual string str(const string indent="") const=0;

    friend ostream& operator<<(ostream& stream, const GirrepObj& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif 
