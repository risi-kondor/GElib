
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSession
#define _GElibSession

#include "CnineSession.hpp"


namespace GElib{

  class GElibSession{
  public:

    cnine::cnine_session* cnine_session=nullptr;


    GElibSession(){
      cnine_session=new cnine::cnine_session();
    }


    ~GElibSession(){
      cout<<endl<<"Shutting down GElib."<<endl;
      delete cnine_session;
    }
    
  };

}


#endif 
