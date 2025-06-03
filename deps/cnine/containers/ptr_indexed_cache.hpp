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


#ifndef _ptr_indexed_cache
#define _ptr_indexed_cache

#include "Cnine_base.hpp"
#include "observable.hpp"
//#include "ptr_pair_indexed_cache.hpp"


namespace cnine{


  template<typename KEY, typename OBJ>
  class ptr_indexed_cache: public unordered_map<KEY*,OBJ>{
  public:

    using unordered_map<KEY*,OBJ>::insert;
    using unordered_map<KEY*,OBJ>::find;
    using unordered_map<KEY*,OBJ>::erase;


    std::function<OBJ(const KEY&)> make_obj;
    observer<KEY> observers;
    
    ~ptr_indexed_cache(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    ptr_indexed_cache():
      make_obj([](const KEY& x){return OBJ();}),
      observers([this](KEY* p){erase(p);}){}

    ptr_indexed_cache(std::function<OBJ(const KEY&)> _make_obj):
      make_obj(_make_obj),
      observers([this](KEY* p){erase(p);}){}


  public: // ---- Access -------------------------------------------------------------------------------------


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

    OBJ& operator()(KEY* keyp){
      auto it=find(keyp);
      if(it!=unordered_map<KEY*,OBJ>::end()) 
	return it->second;
      observers.add(keyp);
      auto p=insert({keyp,make_obj(*keyp)});
      return p.first->second;
    }

  };


}

#endif 

