/*
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


#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Ltensor.hpp"

using namespace cnine;


template<typename TYPE>
class Base{
public:

  TYPE& foo(){
    return static_cast<TYPE&>(*this);
  }

  string repr(){
    return "Base";
  }

};


class SubBase: public Base<SubBase>{
  public:

  SubBase(){}

  SubBase(const Base<SubBase>& x):
    Base<SubBase>(x){}
};


class Derived: public Base<Derived>{
public:
  
  Derived(){}

  Derived(const Base<Derived>& x):
    Base<Derived>(x){}

  Derived(const SubBase& x):
    Base<Derived>(reinterpret_cast<const Base<Derived>&>(x)){}

  string repr(){
    return "Derived";
  }

};




int main(int argc, char** argv){

  cnine_session session;

  SubBase A;
  cout<<A.repr()<<endl;

  Derived B;
  cout<<B.repr()<<endl;
  cout<<B.foo().repr()<<endl;

  Derived C(A);
  cout<<C.repr()<<endl;

}



