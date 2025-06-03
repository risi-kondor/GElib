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


#ifndef _ptr_arg_indexed_cache
#define _ptr_arg_indexed_cache

#include "Cnine_base.hpp"
#include "observable.hpp"

#include "ptr_indexed_cache.hpp" //to avoid duplicate defn of hash fn


namespace cnine{


  template<typename KEY, typename ARG, typename OBJ>
  class ptr_arg_indexed_cache: public unordered_map<std::pair<KEY*,ARG>,OBJ>{
  public:

    typedef std::pair<KEY*,ARG> KEYS;
    typedef unordered_map<KEYS,OBJ> BASE;

    using BASE::insert;
    using BASE::find;
    using BASE::erase;


    std::function<OBJ(const KEY&, const ARG&)> make_obj;
    observer<KEY> observers;
    std::unordered_map<KEY*,std::set<ARG> > lookup0;
    ARG default_arg;

    ~ptr_arg_indexed_cache(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    ptr_arg_indexed_cache():
      make_obj([](const KEY& x, const ARG& y){return OBJ();}),
      observers([this](KEY* p){erase0(p);}){}

    ptr_arg_indexed_cache(std::function<OBJ(const KEY&, const ARG&)> _make_obj):
      make_obj(_make_obj),
      observers([this](KEY* p){erase0(p);}){}

    ptr_arg_indexed_cache(std::function<OBJ(const KEY&, const ARG&)> _make_obj, const ARG& _default):
      make_obj(_make_obj),
      observers([this](KEY* p){erase0(p);}),
      default_arg(_default){}

  private:

    void erase0(KEY* x){
      for(auto y:lookup0[x])
	erase(make_pair(x,y));
      lookup0.erase(x);
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


    bool contains(const KEY& key, const ARG& arg){
      return BASE::find(pair<KEY*,ARG>(&const_cast<KEY&>(key),arg))!=unordered_map<KEYS,OBJ>::end();
    }

    OBJ& insert(const KEY& key, const ARG& arg, const OBJ& val){
      //KEY* keyp=&const_cast<KEY&>(key);
      KEY* keyp=&unconst(key);
      observers.add(keyp);
      lookup0[keyp].insert(arg);
      auto p=insert({make_pair(keyp,arg),val});
      return p.first->second;
    }

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

  };


}

namespace std{

  /*
  template<typename IX1, typename IX2> // duplicate! 
  struct hash<pair<IX1,IX2> >{
  public:
    size_t operator()(const pair<IX1,IX2>& x) const{
      size_t h=hash<IX1>()(x.first);
      h=(h<<1)^hash<IX2>()(x.second);
      return h;
    }
  };
  */
  
}

#endif 

