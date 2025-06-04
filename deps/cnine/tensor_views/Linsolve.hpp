/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineLinsolve
#define _CnineLinsolve

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor1_view.hpp"
#include "TensorView.hpp" // Changed from RtensorObj.hpp

namespace cnine{

#ifdef _WITH_EIGEN
  // Corrected second parameter to Rtensor1_view, return type to TensorView<float>
  extern TensorView<float> eigen_linsolve(const Rtensor2_view& A, const Rtensor1_view& b);
#endif


  class Linsolve{
  public:

    typedef TensorView<float> rtensor; // Changed from RtensorObj

    rtensor operator()(const Rtensor2_view& A, const Rtensor1_view& b){
#ifdef _WITH_EIGEN
      // The extern declaration above is now the one from EigenRoutines.cpp
      return eigen_linsolve(A,b);
#endif
    }

    
  };

}

#endif 
