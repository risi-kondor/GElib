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


#ifndef _ptr_triple_indexed_cache
#define _ptr_triple_indexed_cache

#include "Cnine_base.hpp"
#include "observable.hpp"


namespace cnine{


  template<typename KEY0, typename KEY1, typename KEY2, typename OBJ>
  class ptr_triple_indexed_cache: public unordered_map<std::tuple<KEY0*,KEY1*,KEY2*>,OBJ>{
  public:

    typedef std::tuple<KEY0*,KEY1*,KEY2*> KEYS;

    using unordered_map<KEYS,OBJ>::size;
    using unordered_map<KEYS,OBJ>::insert;
    using unordered_map<KEYS,OBJ>::find;
    using unordered_map<KEYS,OBJ>::erase;


    std::function<OBJ(const KEY0&, const KEY1&, const KEY2&)> make_obj;
    observer<KEY0> observers0;
    observer<KEY1> observers1;
    observer<KEY2> observers2;
    std::unordered_map<KEY0*,std::set<KEYS> > lookup0;
    std::unordered_map<KEY1*,std::set<KEYS> > lookup1;
    std::unordered_map<KEY2*,std::set<KEYS> > lookup2;

    ~ptr_triple_indexed_cache(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    ptr_triple_indexed_cache():
      make_obj([](const KEY0& x0, const KEY1& x1, const KEY2& x2){return OBJ();}),
      observers0([this](KEY0* p){erase0(p);}),
      observers1([this](KEY1* p){erase1(p);}),
      observers2([this](KEY2* p){erase2(p);}){}

    ptr_triple_indexed_cache(std::function<OBJ(const KEY0&, const KEY1&, const KEY2&)> _make_obj):
      make_obj(_make_obj),
      observers0([this](KEY0* p){erase0(p);}),
      observers1([this](KEY1* p){erase1(p);}),
      observers2([this](KEY2* p){erase2(p);}){}


  private:

    void erase0(KEY0* x){
      auto it=lookup0.find(x);
      if(it==lookup0.end()) return;
      for(auto& p:it->second){
	erase(p);
	lookup1[std::get<1>(p)].erase(p);
	lookup2[std::get<2>(p)].erase(p);
      }
      lookup0.erase(it);
    }

    void erase1(KEY1* x){
      auto it=lookup1.find(x);
      if(it==lookup1.end()) return;
      for(auto& p:it->second){
	erase(p);
	lookup0[std::get<0>(p)].erase(p);
	lookup2[std::get<2>(p)].erase(p);
      }
      lookup1.erase(it);
    }

    void erase2(KEY2* x){
      auto it=lookup2.find(x);
      if(it==lookup2.end()) return;
      for(auto& p:it->second){
	erase(p);
	lookup0[std::get<0>(p)].erase(p);
	lookup1[std::get<1>(p)].erase(p);
      }
      lookup2.erase(it);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ operator()(KEY0& key0, KEY1& key1, KEY2& key2){
      return (*this)(&key0,&key1,&key2);
    }

    OBJ operator()(const KEY0& key0, const KEY1& key1, const KEY2& key2){
      return (*this)(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1),&const_cast<KEY2&>(key2));
    }

    OBJ operator()(shared_ptr<KEY0> keyp0, shared_ptr<KEY1> keyp1, shared_ptr<KEY2> keyp2){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      auto& key2=*keyp2;
      return (*this)(&key0,&key1,&key2);
    }

    OBJ operator()(shared_ptr<const KEY0> keyp0, shared_ptr<const KEY1> keyp1, shared_ptr<const KEY2> keyp2){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      auto& key2=*keyp2;
      return (*this)(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1),&const_cast<KEY2&>(key2));
    }

    OBJ& operator()(KEY0* keyp0, KEY1* keyp1, KEY2* keyp2){
      auto p=make_tuple(keyp0,keyp1,keyp2);
      auto it=find(p);
      if(it!=unordered_map<KEYS,OBJ>::end()) 
	return it->second;
      observers0.add(keyp0);
      observers1.add(keyp1);
      observers2.add(keyp2);
      lookup0[keyp0].insert(p);
      lookup1[keyp1].insert(p);
      lookup2[keyp2].insert(p);
      auto it2=insert({p,make_obj(*keyp0,*keyp1,*keyp2)});
      return it2.first->second;
    }

    void insert(KEY0* keyp0, KEY1* keyp1, KEY2* keyp2, const OBJ& x){
      auto p=make_tuple(keyp0,keyp1,keyp2);
      observers0.add(keyp0);
      observers1.add(keyp1);
      observers2.add(keyp2);
      lookup0[keyp0].insert(p);
      lookup1[keyp1].insert(p);
      lookup2[keyp2].insert(p);
      auto it2=insert({p,x});
    }

    size_t rmemsize() const{
      size_t t=0;
      for(auto& p:*this)
	t+=p.second->rmemsize();
      return t;
    }

  };

  template<typename KEY0, typename KEY1, typename KEY2, typename ARG, typename OBJ>
  class ptr_triple_arg_indexed_cache: public unordered_map<std::tuple<KEY0*,KEY1*,KEY2*,ARG>,OBJ>{
  public:

    typedef std::tuple<KEY0*,KEY1*,KEY2*,ARG> KEYS;

    using unordered_map<KEYS,OBJ>::size;
    using unordered_map<KEYS,OBJ>::insert;
    using unordered_map<KEYS,OBJ>::find;
    using unordered_map<KEYS,OBJ>::erase;


    std::function<OBJ(const KEY0&, const KEY1&, const KEY2&, const ARG&)> make_obj;
    observer<KEY0> observers0;
    observer<KEY1> observers1;
    observer<KEY2> observers2;
    std::unordered_map<KEY0*,std::set<KEYS> > lookup0;
    std::unordered_map<KEY1*,std::set<KEYS> > lookup1;
    std::unordered_map<KEY2*,std::set<KEYS> > lookup2;

    ~ptr_triple_arg_indexed_cache(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    ptr_triple_arg_indexed_cache():
      make_obj([](const KEY0& x0, const KEY1& x1, const KEY2& x2, const ARG& arg){return OBJ();}),
      observers0([this](KEY0* p){erase0(p);}),
      observers1([this](KEY1* p){erase1(p);}),
      observers2([this](KEY2* p){erase2(p);}){}

    ptr_triple_arg_indexed_cache(std::function<OBJ(const KEY0&, const KEY1&, const KEY2&, const ARG&)> _make_obj):
      make_obj(_make_obj),
      observers0([this](KEY0* p){erase0(p);}),
      observers1([this](KEY1* p){erase1(p);}),
      observers2([this](KEY2* p){erase2(p);}){}


  private:

    void erase0(KEY0* x){
      auto it=lookup0.find(x);
      if(it==lookup0.end()) return;
      for(auto& p:it->second){
	erase(p);
	lookup1[std::get<1>(p)].erase(p);
	lookup2[std::get<2>(p)].erase(p);
      }
      lookup0.erase(it);
    }

    void erase1(KEY1* x){
      auto it=lookup1.find(x);
      if(it==lookup1.end()) return;
      for(auto& p:it->second){
	erase(p);
	lookup0[std::get<0>(p)].erase(p);
	lookup2[std::get<2>(p)].erase(p);
      }
      lookup1.erase(it);
    }

    void erase2(KEY2* x){
      auto it=lookup2.find(x);
      if(it==lookup2.end()) return;
      for(auto& p:it->second){
	erase(p);
	lookup0[std::get<0>(p)].erase(p);
	lookup1[std::get<1>(p)].erase(p);
      }
      lookup2.erase(it);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    OBJ operator()(KEY0& key0, KEY1& key1, KEY2& key2, const ARG& arg){
      return (*this)(&key0,&key1,&key2,arg);
    }

    OBJ operator()(const KEY0& key0, const KEY1& key1, const KEY2& key2, const ARG& arg){
      return (*this)(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1),&const_cast<KEY2&>(key2),arg);
    }

    OBJ operator()(shared_ptr<KEY0> keyp0, shared_ptr<KEY1> keyp1, shared_ptr<KEY2> keyp2, const ARG& arg){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      auto& key2=*keyp2;
      return (*this)(&key0,&key1,&key2,arg);
    }

    OBJ operator()(shared_ptr<const KEY0> keyp0, shared_ptr<const KEY1> keyp1, shared_ptr<const KEY2> keyp2, const ARG& arg){
      auto& key0=*keyp0;
      auto& key1=*keyp1;
      auto& key2=*keyp2;
      return (*this)(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1),&const_cast<KEY2&>(key2),arg);
    }

    OBJ& operator()(KEY0* keyp0, KEY1* keyp1, KEY2* keyp2, const ARG& arg){
      auto p=make_tuple(keyp0,keyp1,keyp2,arg);
      auto it=find(p);
      if(it!=unordered_map<KEYS,OBJ>::end()) 
	return it->second;
      observers0.add(keyp0);
      observers1.add(keyp1);
      observers2.add(keyp2);
      lookup0[keyp0].insert(p);
      lookup1[keyp1].insert(p);
      lookup2[keyp2].insert(p);
      auto it2=insert({p,make_obj(*keyp0,*keyp1,*keyp2,arg)});
      return it2.first->second;
    }

    bool contains(const KEY0& key0, const KEY1& key1, const KEY2& key2, const ARG& arg){
      auto p=make_tuple(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1),&const_cast<KEY2&>(key2),arg);
      return find(p)!=unordered_map<KEYS,OBJ>::end();
    }

    void insert(const KEY0& key0, const KEY1& key1, const KEY2& key2, const ARG& arg, const OBJ& obj){
      insert(&const_cast<KEY0&>(key0),&const_cast<KEY1&>(key1),&const_cast<KEY2&>(key2),arg,obj);
    }

    void insert(KEY0* keyp0, KEY1* keyp1, KEY2* keyp2, const ARG& arg, const OBJ& x){
      auto p=make_tuple(keyp0,keyp1,keyp2,arg);
      observers0.add(keyp0);
      observers1.add(keyp1);
      observers2.add(keyp2);
      lookup0[keyp0].insert(p);
      lookup1[keyp1].insert(p);
      lookup2[keyp2].insert(p);
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

  template<typename IX1, typename IX2, typename IX3>
  struct hash<tuple<IX1,IX2,IX3> >{
  public:
    size_t operator()(const tuple<IX1,IX2,IX3>& x) const{
      size_t h=hash<IX1>()(get<0>(x));
      h=(h<<1)^hash<IX2>()(get<1>(x));
      h=(h<<1)^hash<IX3>()(get<2>(x));
      return h;
    }
  };

  template<typename IX1, typename IX2, typename IX3, typename IX4>
  struct hash<tuple<IX1,IX2,IX3,IX4> >{
  public:
    size_t operator()(const tuple<IX1,IX2,IX3,IX4>& x) const{
      size_t h=hash<IX1>()(get<0>(x));
      h=(h<<1)^hash<IX2>()(get<1>(x));
      h=(h<<1)^hash<IX3>()(get<2>(x));
      h=(h<<1)^hash<IX4>()(get<3>(x));
      return h;
    }
  };

}


#endif 


