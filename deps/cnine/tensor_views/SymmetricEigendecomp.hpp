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


#ifndef _CnineSymmetricEigendecomp
#define _CnineSymmetricEigendecomp

#include "Cnine_base.hpp"
#include "Gstrides.hpp"
#include "Gtensor.hpp"

#include "Rtensor1_view.hpp"
#include "TensorView.hpp" // Changed from RtensorObj.hpp

namespace cnine{

#ifdef _WITH_EIGEN
  extern pair<TensorView<float>,TensorView<float>> eigen_eigendecomp(const Rtensor2_view& x);
#endif


  class SymmetricEigendecomp{
  public:

    typedef TensorView<float> rtensor; // Changed from RtensorObj

    TensorView<float> U; // Changed from rtensor
    TensorView<float> D; // Changed from rtensor

    SymmetricEigendecomp(const Rtensor2_view& x){
#ifdef _WITH_EIGEN
      auto p=eigen_eigendecomp(x);
      U=p.first; // U and D are now TensorView<float>
      D=p.second;
#endif
    }

    
  };

}

#endif 
