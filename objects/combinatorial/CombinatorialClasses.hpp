
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _CombinatorialClasses
#define _CombinatorialClasses

#include "GElib_base.hpp"
#include "CombinatorialBank.hpp"

#include "IntegerPartition.hpp"
#include "YoungTableau.hpp"

#include "IntegerPartitions.hpp"
#include "YoungTableaux.hpp"


namespace GElib{

  class CombinatorialClasses{
  public:

    CombinatorialClasses(){
      if(_snbank){cout<<"Only one CombinatorialClasses object can be defined."<<endl; return;}
      _combibank=new CombinatorialBank();
    }
    
    ~CombinatorialClasses(){
      delete _combibank;
    }

  };

}

#endif 
