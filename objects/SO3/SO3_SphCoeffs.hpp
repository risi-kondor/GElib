/*
 * This file is part of GElib, a C++/CUDA library for group
 * equivariant tensor operations. 
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


#ifndef _SO3_SphCoeffs
#define _SO3_SphCoeffs

#include "Rtensor.hpp"


namespace GElib{


  class SO3_SPHgen{
  public:

    using rtensor=cnine::RtensorA;

    RtensorA coeffs;

    SO3_SPHgen(){
    }


  };

}

