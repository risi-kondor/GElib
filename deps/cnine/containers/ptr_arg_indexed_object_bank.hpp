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


#ifndef _ptr_arg_indexed_object_bank
#define _ptr_arg_indexed_object_bank

#include "Cnine_base.hpp"
#include "observable.hpp"


namespace cnine{


  template<typename KEY, typename ARG, typename OBJ>
  class ptr_arg_indexed_object_bank: public unordered_map<std::pair<KEY*,ARG>,OBJ>{
  public:

    typedef std::pair<KEY*,ARG> KEYS;
    typedef unordered_map<KEYS,OBJ> BASE;

    using BASE::size;
    using BASE::insert;
    using BASE::find;
    using BASE::erase;


    std::function<OBJ(const KEY&, const ARG&)> make_obj;
    observer<KEY> observers;
    std::unordered_map<KEY*,std::set<ARG> > lookup0;
    ARG default_arg;

    ~ptr_arg_indexed_object_bank(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    ptr_arg_indexed_object_bank():
      make_obj([](const KEY& x, const ARG& y){cout<<"empty object in bank"<<endl; return OBJ();}),
      observers([this](KEY* p){erase(p);}){}

    ptr_arg_indexed_object_bank(std::function<OBJ(const KEY&, const ARG&)> _make_obj):
      make_obj(_make_obj),
      observers([this](KEY* p){erase0(p);}){}

    ptr_arg_indexed_object_bank(std::function<OBJ(const KEY&, const ARG&)> _make_obj, const ARG& _default):
      make_obj(_make_obj),
      observers([this](KEY* p){erase0(p);}),
      default_arg(_default){}

  private:

    void erase0(KEY* x){
      for(auto y:lookup0[x])
	erase(make_pair(x,y));
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ& operator()(KEY* keyp, const ARG& arg){
      auto it=find(make_pair(keyp,arg));
      if(it!=unordered_map<KEYS,OBJ>::end()) 
	return it->second;
      observers.add(keyp);
      lookup0[keyp].insert(arg);
      auto p=insert({make_pair(keyp,arg),make_obj(*keyp,arg)});
      return p.first->second;
    }

    OBJ& operator()(KEY* keyp){
      return (*this)(keyp,default_arg);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ operator()(KEY& key, const ARG& arg){
      return (*this)(&key,arg);
    }

    OBJ operator()(const KEY& key, const ARG& arg){
      return (*this)(&const_cast<KEY&>(key),arg);
    }

    OBJ operator()(shared_ptr<KEY> keyp, const ARG& arg){
      auto& key=*keyp;
      return (*this)(&key,arg);
    }

    OBJ operator()(shared_ptr<const KEY> keyp, const ARG& arg){
      const auto& key=*keyp;
      return (*this)(&const_cast<KEY&>(key,arg));
    }


    /*
    OBJ operator()(KEY& key){
      return (*this)(&key);
    }

    OBJ operator()(const KEY& key){
      return (*this)(&const_cast<KEY&>(key));
    }

    OBJ operator()(shared_ptr<KEY> keyp){
      auto& key=*keyp;
      return (*this)(&key);
    }

    OBJ operator()(shared_ptr<const KEY> keyp){
      const auto& key=*keyp;
      return (*this)(&const_cast<KEY&>(key));
    }
    */


  };


}

#endif 

