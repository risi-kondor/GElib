// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibLog
#define _GElibLog

#include <fstream>
#include <chrono>
#include <ctime>

namespace GElib{

  class GElibLog{
  public:

    ofstream ofs;

    GElibLog(const string filename="GElib.log"){
      ofs.open(filename);
      auto time = std::chrono::system_clock::now();
      std::time_t timet = std::chrono::system_clock::to_time_t(time);
      ofs<<"GElib log opened "<<std::ctime(&timet)<<endl;
    }

    ~GElibLog(){
      ofs.close();
    }

    void operator()(const string msg){
      //auto time = std::chrono::system_clock::now();
      //std::time_t timet = std::chrono::system_clock::to_time_t(time);
      std::time_t timet = std::time(nullptr);
      char os[30];
      strftime(os,30,"%H:%M:%S ",std::localtime(&timet));
      //ofs<<std::asctime(std::localtime(&timet))<<"  ";
      ofs<<os<<msg<<endl;
    }

  };


}

#endif 
