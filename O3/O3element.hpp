/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2025, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GElibO3element
#define _GElibO3element

#include "GElib_base.hpp"
#include "Gelement.hpp"
#include "ColumnSpace.hpp"
#include "TensorView_functions.hpp"


namespace GElib{

  template<typename TYPE>
  class O3element: public Gelement, public cnine::TensorView<TYPE>{ //public GelementRealMx<SO3element>{
  public:

    //typedef GelementRealMx<SO3element> BASE;
    typedef cnine::TensorView<TYPE> BASE;
    using BASE::BASE;
    using BASE::str; 

    O3element(const BASE& x):
      BASE(x){}


  public: // ---- Named constructors -------------------------------------------------------------------------


    static O3element identity(){
      return BASE::identity(3);}

    static O3element random(){
      cnine::TensorView<TYPE> A(cnine::dims(3,3),4,0);
      cnine::TensorView<TYPE> B((cnine::ColumnSpace<float>(A)()));
      return O3element(B);
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    O3element operator*(const O3element& y) const{
      cnine::TensorView<TYPE> R({3,3},0,0);
      R.add_mprod(*this,y);
      return R;
    }

    O3element inv() const{
      return this->transp();
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string repr() const{
      ostringstream oss;
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const O3element& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
