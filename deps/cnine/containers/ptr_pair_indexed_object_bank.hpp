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


#ifndef _ptr_pair_indexed_object_bank
#define _ptr_pair_indexed_object_bank

#include "Cnine_base.hpp"
#include "observable.hpp"


namespace cnine{


  template<typename KEY0, typename KEY1, typename OBJ>
  class ptr_pair_indexed_object_bank: public unordered_map<std::pair<KEY0*,KEY1*>,OBJ>{
  public:

    typedef std::pair<KEY0*,KEY1*> KEYS;

    using unordered_map<KEYS,OBJ>::size;
    using unordered_map<KEYS,OBJ>::insert;
    using unordered_map<KEYS,OBJ>::find;
    using unordered_map<KEYS,OBJ>::erase;


    std::function<OBJ(const KEY0&, const KEY1&)> make_obj;
    observer<KEY0> observers0;
    observer<KEY1> observers1;
    std::unordered_map<KEY0*,std::set<KEY1*> > lookup0;
    std::unordered_map<KEY1*,std::set<KEY0*> > lookup1;

    ~ptr_pair_indexed_object_bank(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    ptr_pair_indexed_object_bank():
      make_obj([](const KEY0& x0, const KEY1& x1){cout<<"empty object in bank"<<endl; return OBJ();}),
      observers0([this](KEY0* p){erase0(p);}),
      observers1([this](KEY1* p){erase1(p);}){}

    ptr_pair_indexed_object_bank(std::function<OBJ(const KEY0&, const KEY1&)> _make_obj):
      make_obj(_make_obj),
      observers0([this](KEY0* p){erase0(p);}),
      observers1([this](KEY1* p){erase1(p);}){}


  private:

    void erase0(KEY0* x){
      auto it=lookup0.find(x);
      if(it==lookup0.end()) return;
      for(auto y:it->second){
	erase(make_pair(x,y));
	lookup1[y].erase(x);
      }
      lookup0.erase(it);
    }

    void erase1(KEY1* y){
      auto it=lookup1.find(y);
      if(it==lookup1.end()) return;
      for(auto x:it->second){
	erase(make_pair(x,y));
	lookup0[x].erase(y);
      }
      lookup1.erase(it);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ operator()(KEY0& key0, KEY1& key1){
      return (*this)(&key0,&key1);
    }

    OBJ operator()(const KEY0& key0, const KEY1& key1){
      return (*this)(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1));
    }

    OBJ operator()(shared_ptr<KEY0> keyp0, shared_ptr<KEY1> keyp1){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      return (*this)(&key0,&key1);
    }

    OBJ operator()(shared_ptr<const KEY0> keyp0, shared_ptr<const KEY1> keyp1){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      return (*this)(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1));
    }

    OBJ& operator()(KEY0* keyp0, KEY1* keyp1){
      auto it=find(make_pair(keyp0,keyp1));
      if(it!=unordered_map<KEYS,OBJ>::end()) 
	return it->second;
      observers0.add(keyp0);
      observers1.add(keyp1);
      lookup0[keyp0].insert(keyp1);
      lookup1[keyp1].insert(keyp0);
      auto p=insert({make_pair(keyp0,keyp1),make_obj(*keyp0,*keyp1)});
      return p.first->second;
    }

    bool contains(const shared_ptr<KEY0>& keyp0, const shared_ptr<KEY1>& keyp1){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      return contains(&key0,&key1);
    }

    bool contains(const KEY0& key0, const KEY1& key1){
      auto p=make_pair(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1));
      return find(p)!=unordered_map<KEYS,OBJ>::end();
    }

    void insert(const shared_ptr<KEY0>& keyp0, const shared_ptr<KEY1>& keyp1, const OBJ& x){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      insert(&key0,&key1,x);
    }

    void insert(const KEY0& key0, const KEY1& key1, const OBJ& x){
      insert(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1),x);
    }

    void insert(KEY0* keyp0, KEY1* keyp1, const OBJ& x){
      auto p=make_pair(keyp0,keyp1);
      observers0.add(keyp0);
      observers1.add(keyp1);
      lookup0[keyp0].insert(keyp1);
      lookup1[keyp1].insert(keyp0);
      auto it2=insert({p,x});
    }

    size_t rmemsize() const{
      size_t t=0;
      for(auto& p:*this)
	t+=p.second->rmemsize();
      return t;
    }

  };



}

namespace std{

  template<typename IX1, typename IX2>
  struct hash<pair<IX1,IX2> >{
  public:
    size_t operator()(const pair<IX1,IX2>& x) const{
      size_t h=hash<IX1>()(x.first);
      h=(h<<1)^hash<IX2>()(x.second);
      return h;
    }
  };

}


#endif 


