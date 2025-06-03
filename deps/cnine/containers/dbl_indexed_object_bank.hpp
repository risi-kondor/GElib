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


#ifndef _dbl_indexed_object_bank
#define _dbl_indexed_object_bank

#include "Cnine_base.hpp"

namespace cnine{

  template<typename KEY1, typename KEY2, typename OBJ>
  class dbl_indexed_object_bank: public unordered_map<KEY2,OBJ*>{
  public:

    using unordered_map<KEY2,OBJ*>::find;


    std::function<OBJ*(const KEY&)> make_obj;
    
    ~dbl_indexed_object_bank(){
      for(auto& p:*this) delete p.second;
    }

    dbl_indexed_object_bank():
      make_obj([](const KEY& x){return nullptr;}){}

    dbl_indexed_object_bank(std::function<OBJ*(const KEY&)> _make_obj):
      make_obj(_make_obj){}


  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ& operator()(const KEY& key){
      auto it=find(key);
      if(it!=unordered_map<KEY,OBJ*>::end()) return *(*this)[key];
      OBJ* new_obj=make_obj(key);
      (*this)[key]=new_obj;
      return *new_obj;
    }

  };

}

#endif 
