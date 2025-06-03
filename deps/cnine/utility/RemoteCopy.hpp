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


#ifndef _cnine_RemoteCopy
#define _cnine_RemoteCopy

#include "Cnine_base.hpp"
#include <unordered_map>

namespace cnine{


  template<typename KEY, typename OBJ> 
  class RemoteCopy{
  public:

    mutable unordered_map<KEY,shared_ptr<OBJ> > map;

    std::function<shared_ptr<OBJ>(const KEY&)> make_obj;

    RemoteCopy():
      make_obj([](const KEY& x){return nullptr;}){}

    RemoteCopy(std::function<shared_ptr<OBJ>(const KEY&)> _make_obj):
      make_obj(_make_obj){}


  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ& operator()(const KEY& key) const{
      auto it=map.find(key);
      if(it!=map.end()) return *map[key];
      auto new_obj=make_obj(key);
      map[key]=new_obj;
      return *new_obj;
    }

    void insert(const KEY& key, const shared_ptr<OBJ>& x){
      map[key]=x;
    }

  };

}

#endif 
