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
    int n_ops=0;
    chrono::time_point<chrono::system_clock> t0;

    LoggedTimer(string _task=""):
      task(_task){
      t0=chrono::system_clock::now();
    }

    LoggedTimer(string _task, const int _ops):
      task(_task){
      t0=chrono::system_clock::now();
      n_ops=_ops;
    }

    ~LoggedTimer(){
      auto elapsed=chrono::duration<double,std::milli>(chrono::system_clock::now()-t0).count();
      if(gelib_log){
	if(n_ops>0) (*gelib_log)(task+" "+to_string(elapsed)+" ms"+" ["+to_string((int)(((float)n_ops)/elapsed/1000.0))+" Mflops]");
	else (*gelib_log)(task+" "+to_string(elapsed)+" ms");
      }
    }

  };


  class CGproductTimer{
  public:

    string task;
    int l1,l2,l;
    int b,n1,n2,dev;
    int n_ops=0;
    chrono::time_point<chrono::system_clock> t0;

    //LoggedTimer(string _task=""):
    //task(_task){
    //t0=chrono::system_clock::now();
    //}

    //LoggedTimer(string _task, const int _ops):
    //task(_task){
    //t0=chrono::system_clock::now();
    //n_ops=_ops;
    //}

    CGproductTimer(const int _l1, const int _l2, const int _l, const int _b, 
      const int _n1, const int _n2, const int _dev, const int _ops):
      l1(_l1),l2(_l2),l(_l),b(_b),n1(_n1),n2(_n2),dev(_dev),n_ops(_ops){ 
      t0=chrono::system_clock::now();
    }

    ~CGproductTimer(){
      auto elapsed=chrono::duration<double,std::milli>(chrono::system_clock::now()-t0).count();
      int Mflops=0;
      if(elapsed>0) Mflops=(((float)n_ops)/elapsed/1000.0);

      if(gelib_log){
	(*gelib_log)("CGproduct("+to_string(l1)+","+to_string(l2)+","+to_string(l)+")[b="+
	  to_string(b)+",n1="+to_string(n1)+",n2="+to_string(n2)+",dev="+to_string(dev)+"] "+
	  to_string(elapsed)+" ms"+" ["+to_string(Mflops)+" Mflops]");

	  gelib_log->ofs2<<l1<<","<<l2<<","<<l<<","<<b<<","<<n1<<","<<n2<<","<<dev<<","<<elapsed<<","<<Mflops<<endl;
	
      }
    }

  };


}

#endif 
