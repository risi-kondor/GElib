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


#ifndef _CnineCallStack
#define _CnineCallStack

#include <chrono>
#include <ctime>
#include "Cnine_base.hpp"


namespace cnine{


  class CallStackEntry{
  public:
    string name;

    CallStackEntry(const string s):
      name(s){}

    string str() const{
      return name;
    }

  };


  class CallStack: vector<CallStackEntry>{
  public:

    typedef vector<CallStackEntry> BASE;

    void push(const string& s){
      push_back(s);
    }

    void pop(){
      BASE::pop_back();
    }
      
    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:*this)
	oss<<indent<<p.str()<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CallStack& x){
      stream<<x.str(); return stream;
    }

  };


  extern cnine::CallStack call_stack;


  class tracer{
  public:

    tracer(const string s){
      call_stack.push(s);
    }

    ~tracer(){
      call_stack.pop();
    }

  };

}

#endif
