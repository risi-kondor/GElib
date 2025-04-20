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

#ifndef _GElibSO3element
#define _GElibSO3element

#include "GElib_base.hpp"
#include "Gelement.hpp"
#include "ColumnSpace.hpp"
#include "TensorView_functions.hpp"


namespace GElib{

  template<typename TYPE>
  class SO3element: public Gelement, public cnine::TensorView<TYPE>{ //public GelementRealMx<SO3element>{
  public:

    //typedef GelementRealMx<SO3element> BASE;
    typedef cnine::TensorView<TYPE> BASE;
    using BASE::BASE;
    using BASE::str; 

    SO3element(const BASE& x):
      BASE(x){}


  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3element identity(){
      return BASE::identity(3);}

    static SO3element random(){
      cnine::TensorView<TYPE> A(cnine::dims(3,3),4,0);
      cnine::TensorView<TYPE> B((cnine::ColumnSpace<float>(A)()));
      while(B.dim(1)<3){
	cnine::TensorView<TYPE> A(cnine::dims(3,3),4,0);
	B.reset(cnine::ColumnSpace<float>(A)());
      }
      float det=B(0,0)*(B(1,1)*B(2,2)-B(1,2)*B(2,1));
      det+=B(0,1)*(B(1,2)*B(2,0)-B(1,0)*B(2,2));
      det+=B(0,2)*(B(1,0)*B(2,1)-B(1,1)*B(2,0));
      if(det>0) return SO3element(B);
      else return SO3element(B*(-1.0));
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    SO3element operator*(const SO3element& y) const{
      cnine::TensorView<TYPE> R({3,3},0,0);
      R.add_mprod(*this,y);
      return R;
    }

    SO3element inv() const{
      return this->transp();
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string repr() const{
      ostringstream oss;
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const SO3element& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
