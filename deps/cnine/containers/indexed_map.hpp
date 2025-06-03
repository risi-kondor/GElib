/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _indexed_map
#define _indexed_map

#include "Cnine_base.hpp"
#include "GenericIterator.hpp"


namespace cnine{

  template<typename KEY,typename OBJ>
  class indexed_map{
  public:

    class iterator: public GenericIterator<indexed_map,OBJ*>{
    public:
      using GenericIterator<indexed_map,OBJ*>::GenericIterator;
    };


    vector<OBJ*> v;
    mutable map<KEY,OBJ*> _map;
    
    ~indexed_map(){
      for(auto p:v) 
	delete p;
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    indexed_map& operator=(indexed_map&& x){
      v=x.v;
      x.v.clear();
      _map=x._map;
      x._map.clear();
    }


  public: // ---- Access -------------------------------------------------------------------------------------

    int size() const{
      return v.size();
    }

    OBJ* operator[](const int i) const{
      return v[i];
    }

    OBJ* operator[](const KEY& key) const{
      return _map[key];
    }

    void insert(const KEY& key, OBJ* obj){
       v.push_back(obj);
      _map[key]=obj;
    }

    bool exists(const KEY& key) const{
      return _map.find(key)!=_map.end();
    }

    iterator begin() const{
      return iterator(this);
    }

    iterator end() const{
      return iterator(this,size());
    }

    void clear(){
      v.clear();
      _map.clear();
    }

    void wipe(){
      for(auto p:v) 
	delete p;
      v.clear();
      _map.clear();
    }

  };


  template<typename KEY,typename OBJ>
  class indexed_mapB{
  public:

    class iterator: public GenericIterator<indexed_mapB,OBJ*>{
    public:
      using GenericIterator<indexed_mapB,OBJ*>::GenericIterator;
    };


    vector<OBJ*> v;
    mutable map<KEY,OBJ*> _map;
    
    ~indexed_mapB(){
      for(auto p:v) 
	delete p;
    }

    indexed_mapB(){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    indexed_mapB(const indexed_mapB& x){
      for(auto p:x.v){
	push_back(new OBJ(*p));
      }
    }

    indexed_mapB(indexed_mapB&& x){
      v=x.v;
      _map=x._map;
      x.v.clear();
      x._map.clear();
    }

    indexed_mapB& operator=(const indexed_mapB& x){
      wipe();
      for(auto p:x.v){
	push_back(new OBJ(*p));
      }
      return *this;
    }

    indexed_mapB& operator=(indexed_mapB&& x){
      wipe();
      v=x.v;
      _map=x._map;
      x.v.clear();
      x._map.clear();
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return v.size();
    }

    OBJ* operator[](const int i) const{
      return v[i];
    }

    OBJ* operator[](const KEY& key) const{
      return _map[key];
    }

    void insert(const KEY& key, OBJ* obj){
      v.push_back(obj);
      _map[key]=obj;
    }

    void push_back(OBJ* obj){
      v.push_back(obj);
      _map[obj->key()]=obj;
    }

    bool exists(const KEY& key) const{
      return _map.find(key)!=_map.end();
    }

    iterator begin() const{
      return iterator(this);
    }

    iterator end() const{
      return iterator(this,size());
    }

    void clear(){
      v.clear();
      _map.clear();
    }

    void wipe(){
      for(auto p:v) 
	delete p;
      v.clear();
      _map.clear();
    }

  };

}

#endif
