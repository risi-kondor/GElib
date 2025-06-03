/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _observer
#define _observer

#include "Cnine_base.hpp"
#include <set>
#include <unordered_map>


namespace cnine{


  template<typename TARGET>
  class observer;


  template<typename TARGET>
  class observer_hook{
  public:

    TARGET* owner;
    set<observer<TARGET>*> observers;

    observer_hook(TARGET* _owner):
      owner(_owner){}

    ~observer_hook(){
      for(auto p:observers)
	p->killed_signal(owner);
    }

    void attach(const observer<TARGET>* x){
      if(observers.find(x)!=observers.end()) return;
      observers.insert(x);
    }

    void detach(const observer<TARGET>* x){
      if(observers.find(x)==observers.end()) return;
      cout<<"Observing object "<<x->name<<" detached."<<endl;
      observers.erase(x);
    }

  };



  template<typename TARGET>
  class observer{
  public:

    set<TARGET*> targets;

    ~observer(){
      for(auto p:targets)
	p->detach(*this);
    }
    
    void add(const TARGET* x){
      if(targets.find(x)!=targets.end()) return;
      x->observers.attach(this);
      targets.insert(x);
    }

    void killed_signal(TARGET* x){
      cout<<"Observed object "<<x->name<<" deleted."<<endl;
      CNINE_ASSRT(targets.find(x)!=targets.end());
      targets.erase(x);
    }

  };



}

#endif 
