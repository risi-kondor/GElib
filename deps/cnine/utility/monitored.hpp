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


#ifndef _cnine_watched
#define _cnine_watched

#include "Cnine_base.hpp"
#include "observable.hpp"


namespace cnine{

  template<typename OBJ> 
  class obj_monitor; 


  template<typename OBJ> 
  class monitored: public observable<monitored<OBJ> >{
  public:


    obj_monitor<OBJ>* _obj_monitor=nullptr;

    shared_ptr<OBJ> obj;

    std::function<shared_ptr<OBJ>()> make_obj;

    //monitored():
    //observable<monitored<OBJ> >(this),
    //make_obj([](){return shared_ptr<OBJ>();}){}

    monitored(std::function<shared_ptr<OBJ>()> _make_obj):
      observable<monitored<OBJ> >(this),
      make_obj(_make_obj){}

    monitored(obj_monitor<OBJ>& __obj_monitor, std::function<shared_ptr<OBJ>()> _make_obj):
      observable<monitored<OBJ> >(this),
      _obj_monitor(&__obj_monitor),
      make_obj(_make_obj){}

    ~monitored(){
    }


  public: // ---- Copying ----------------------------------------------------------------------------------


    monitored(const monitored& x):
      observable<monitored<OBJ> >(this),
      _obj_monitor(x._obj_monitor),
      obj(x.obj){}


  public: // ---- Access ----------------------------------------------------------------------------------


    operator OBJ&(){
      return (*this)();
    }

    OBJ& operator()(){
      if(obj.get()==nullptr) make();
      return *obj;
    }

    shared_ptr<OBJ> shared(){
      if(obj.get()==nullptr) make();
      return obj;
    }

    void make(){
      obj=make_obj();
      if(_obj_monitor) 
	_obj_monitor->add(this);
    }

    string str() const{
      if(obj.get()==nullptr) return "";
      return ""; //to_string(*obj);
    }

  };



  template<typename OBJ>
  class obj_monitor{
  public:

    observer<monitored<OBJ> > _monitored;

    void add(monitored<OBJ>* x){
      _monitored.add(x);
    }

    string str() const{
      ostringstream oss;
      oss<<"[";
      for(auto p:_monitored.targets)
	oss<<p->str()<<",";
      oss<<"]";
      return oss.str();
   }

    friend ostream& operator<<(ostream& stream, const obj_monitor& x){
      stream<<x.str(); return stream;
    }

  };



}


#endif 
