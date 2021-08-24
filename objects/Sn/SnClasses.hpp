
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SnObjects
#define _SnObjects

#include "GElib_base.hpp"
#include "SnBank.hpp"

//#include "IntegerPartition.hpp"
//#include "IntegerPartitions.hpp"
#include "Sn.hpp"


namespace GElib{

  class SnClasses{
  public:

    SnClasses(){
      if(_snbank){cout<<"Only one SnClasses object can be defined."<<endl; return;}
      _snbank=new SnBank();
    }
    
    ~SnClasses(){
      delete _snbank;
    }

  };

}

#endif 
