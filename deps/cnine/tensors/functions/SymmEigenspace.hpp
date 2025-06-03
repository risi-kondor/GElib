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

#ifndef _SymmEigenspace
#define _SymmEigenspace

//#include "../TensorFunctions.hpp"
#include "ComplementSpace.hpp"


namespace cnine{

  template<typename TYPE>
  class SymmEigenspace{
  public:

    TensorView<TYPE> T;
    int ncols=0;

    SymmEigenspace(const TensorView<TYPE> X, const TYPE lambda){
      CNINE_ASSRT(X.ndims()==2);
      CNINE_ASSRT(X.dims[0]==X.dims[1]);

      int n=X.dims[0];
      for(int i=0; i<n; i++)
	X.inc(i,i,-lambda);
      T=ComplementSpace<TYPE>(X)();
    }

    TensorView<TYPE> operator()() const{
      return T;
    }

  };

}

#endif 
