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

#ifndef _CnineObservable
#define _CnineObservable

#include "Cnine_base.hpp"
#include <set>
#include <unordered_map>


namespace cnine{


  template<typename TARGET>
  class observer;


  template<typename TARGET>
  class observable{
  public:

    TARGET* owner;
    mutable set<observer<TARGET>*> observers;

    observable(TARGET* _owner):
      owner(_owner){}

    ~observable(){
      for(auto p:observers)
	p->deleted_signal(owner);
    }


  public: // -----------------------------------------


    observable(const observable& x)=delete;

    void attach_observer(observer<TARGET>* x) const{
      if(observers.find(x)!=observers.end()) return;
      observers.insert(x);
    }

    void detach_observer(observer<TARGET>* x) const{
      if(observers.find(x)==observers.end()) return;
      //cout<<x->name<<" no longer observing "<<owner->name<<"."<<endl;
      observers.erase(x);
    }


  };




  template<typename TARGET>
  class observer{
  public:

    set<TARGET*> targets;
    std::function<void(TARGET*)> on_deleted_action;

    ~observer(){
      for(auto p:targets)
	p->detach_observer(this);
    }


  public: // -----------------------------------------

    
    observer(){}

    observer(TARGET* p){
      p->attach_observer(this);
      targets.insert(p);
    }

    observer(const std::function<void(TARGET*)> lambda):
      on_deleted_action(lambda){
    }

    observer(TARGET* p, const std::function<void(TARGET*)> lambda):
      on_deleted_action(lambda){
      p->attach_observer(this);
      targets.insert(p);
    }


  public: // -----------------------------------------


    void add(TARGET* x){
      if(targets.find(x)!=targets.end()) return;
      x->attach_observer(this);
      targets.insert(x);
    }

    void operator()(TARGET* x){
      if(targets.find(x)!=targets.end()) return;
      x->attach_observer(this);
      targets.insert(x);
    }

    void remove(TARGET* x){
      if(targets.find(x)==targets.end()) return;
      x->detach_observer(this);
      targets.erase(x);
    }

    void deleted_signal(TARGET* x){
      //cout<<"Observed object "<<x->name<<" deleted (detected by "<<name<<")"<<endl;
      CNINE_ASSRT(targets.find(x)!=targets.end());
      targets.erase(x);
      if(on_deleted_action) on_deleted_action(x);
    }

  };



  template<typename TARGET, typename PARAM>
  class parametric_observer{
  public:

    unordered_map<TARGET*,PARAM> targets;
    std::function<void(TARGET*,const PARAM&)> on_deleted_action;

    ~parametric_observer(){
      for(auto p:targets)
	p.first->detach_observer(this);
    }


  public: // -----------------------------------------

    
    parametric_observer(){}

    parametric_observer(TARGET* p, const PARAM& x){
      p->attach_observer(this);
      targets[p]=x;
    }

    parametric_observer(const std::function<void(TARGET*, const PARAM&)> lambda):
      on_deleted_action(lambda){
    }

    //parametric_observer(TARGET* p, const std::function<void(TARGET*, const PARAM&)> lambda):
    //on_deleted_action(lambda){
    //p->attach_observer(this);
    //targets[p];
    //}


  public: // -----------------------------------------


    void add(TARGET* x, const PARAM& p){
      if(targets.find(x)!=targets.end()) return;
      x->attach_observer(this);
      targets[x]=p;
    }

    void operator()(TARGET* x, const PARAM& p){
      if(targets.find(x)!=targets.end()) return;
      x->attach_observer(this);
      targets[x]=p;
    }

    void remove(TARGET* x){
      if(targets.find(x)==targets.end()) return;
      x->detach_observer(this);
      targets.erase(x);
    }

    void deleted_signal(TARGET* x){
      //cout<<"Observed object "<<x->name<<" deleted (detected by "<<name<<")"<<endl;
      CNINE_ASSRT(targets.find(x)!=targets.end());
      auto p=targets[x];
      targets.erase(x);
      if(on_deleted_action) on_deleted_action(x,p);
    }

  };



}

#endif 
