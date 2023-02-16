// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibTimer
#define _GElibTimer

#include "GElibLog.hpp"

#include <fstream>
#include <chrono>
#include <ctime>

extern GElib::GElibLog* gelib_log;

namespace GElib{

  class LoggedTimer{
  public:

    string task;
    chrono::time_point<chrono::system_clock> t0;

    LoggedTimer(string _task=""):
      task(_task){
      t0=chrono::system_clock::now();
    }

    ~LoggedTimer(){
      auto elapsed=chrono::duration<double,std::milli>(chrono::system_clock::now()-t0).count();
      if(gelib_log) (*gelib_log)(task+" "+to_string(elapsed)+" ms");
    }

  };


}

#endif 
