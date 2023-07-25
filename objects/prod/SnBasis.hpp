// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SnBasis
#define _SnBasis

#include "Tensor.hpp"
#include "Gtype.hpp"

#include "SnType.hpp"


namespace GElib{

  template<typename TYPE>
  class SnBasis{
  public:

    Snob2::SnType tau;
    cnine::Tensor<TYPE> T;
    
    SnBasis(){}

    SnBasis(const Snob2::SnType& _tau, const cnine::Tensor<TYPE>& _T):
      tau(_tau), T(_T){}


  public: // ---- Operations ---------------------------------------------------------------------------------


    SnBasis conjugate(const cnine::Tensor<TYPE>& x){
      return SnBasis(tau,cnine::transp(x)*T*x);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string repr() const{
      ostringstream oss;
      oss<<"SnBasis<"; 
      return oss.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<T<<endl;
      //if(offsets) oss<<offsets->str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SnBasis& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
