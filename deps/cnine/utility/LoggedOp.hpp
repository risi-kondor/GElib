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


#ifndef _LoggedOp
#define _LoggedOp

#include "Cnine_base.hpp"
#include "CnineLog.hpp"

extern cnine::CnineLog cnine_log;


namespace cnine{

  class LoggedOp{
  public:

    string str;
    int nops=0;
    chrono::time_point<chrono::system_clock> t0;

    LoggedOp(string _str=""):
      str(_str){
      t0=chrono::system_clock::now();
    }

    ~LoggedOp(){
      auto elapsed=chrono::duration<double,std::milli>(chrono::system_clock::now()-t0).count();
      cnine_log("  "+str+": "+to_string(elapsed)+" ms");
      //if(n_ops>0) (*gelib_log)(task+" "+to_string(elapsed)+" ms"+" ["+to_string((int)(((float)n_ops)/elapsed/1000.0))+" Mflops]");
    }


  public: // -------------------------------------------------------------------------------------------------


    LoggedOp(string _str, const int _nops):
      str(_str), nops(_nops){
      t0=chrono::system_clock::now();
    }

    template<typename TYPE1, typename TYPE2, typename TYPE3>
    LoggedOp(string _name, const TYPE1& arg1, const TYPE2& arg2, const TYPE3& arg3):
      LoggedOp(_name+"("+arg1.repr()+", "+arg2.repr()+", "+arg3.repr()+")"){}

  };

}


#endif 
