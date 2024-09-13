
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
//#include "GElibConfig.hpp"
//#include "GElibLog.hpp"

//extern GElib::GElibConfig* gelib_config;
//extern GElib::GElibLog* gelib_log;

namespace GElib{

  class GElibSession{
  public:

    cnine::cnine_session* cnine_session=nullptr;


    GElibSession(const int _nthreads=1){

      #ifdef _WITH_CUDA
      cout<<"Initializing GElib with GPU support."<<endl;
      #else
      cout<<"Initializing GElib without GPU support."<<endl;
      #endif

      cnine_session=new cnine::cnine_session(_nthreads);
      //gelib_config=new GElibConfig();
      //gelib_log=new GElibLog();
    }


    ~GElibSession(){
      cout<<"Shutting down GElib."<<endl;
      //delete gelib_log;
      //delete gelib_config;
      delete cnine_session;
    }
    
  };

}


#endif 
