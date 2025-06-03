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

#ifndef _associative_container
#define _associative_container

#include "Cnine_base.hpp"
#include "GenericIterator.hpp"


namespace cnine{


  template<typename KEY, typename OBJ>
  class associative_container_val_iterator{
  public:
    
    typename map<KEY,OBJ*>::iterator it;

    associative_container_val_iterator(const typename map<KEY,OBJ*>::iterator& _it):
      it(_it){}

    void operator++(){++it;}

    void  operator++(int a){++it;}

    OBJ& operator*(){
      return *it->second; 
    }

    const OBJ& operator*() const{
      return *it->second; 
    }

    bool operator==(const associative_container_val_iterator& x) const{
      return it==x.it;
    }

    bool operator!=(const associative_container_val_iterator& x) const{
      return it!=x.it;
    }

  };


  template<typename KEY, typename OBJ>
  class associative_container_temporary{
  public:

    const KEY& _key;
    OBJ* _val;

    associative_container_temporary(const KEY& __key, OBJ* __val): _key(__key), _val(__val){}
      
  public:

    //operator OBJ&(){
    //return *_val;
    //}

    OBJ& operator*(){
      return *_val;
    }

    OBJ& operator->(){
      return *_val;
    }

    OBJ& val(){
      return *_val;
    }

    const KEY& key(){
      return _key;
    }

  };



  
  template<typename KEY, typename OBJ>
  class associative_container_iterator{
  public:
    
    typedef typename map<KEY,OBJ*>::iterator _iterator;
    typedef associative_container_temporary<KEY,OBJ> temp;

    _iterator it;

    associative_container_iterator(const _iterator& _it):
      it(_it){}

    void operator++(){++it;}

    void  operator++(int a){++it;}

    //pair<KEY,const OBJ&> operator*() const{
    //return pair<KEY,const OBJ&>(it->first,*it->second);
    //}

    int dummy() const{}
      
    temp operator*() const{
      return temp(it->first,it->second); 
    }
      
    temp operator->() const{
      return temp(it->first,it->second); 
    }
      
    //OBJ& operator->() const{
    //return it->second; 
    //}
      
    const KEY key() const{
      return it->first;
    }

    OBJ& val(){
      return *it->second;
    }

    //const KEY& key() const{
    //return it->second;
    //}

    bool operator==(const associative_container_iterator& x) const{
      return it==x.it;
    }

    bool operator!=(const associative_container_iterator& x) const{
      return it!=x.it;
    }

  };


  template<typename KEY,typename OBJ>
  class associative_container_keyval{
  public:

    typedef associative_container_iterator<KEY,OBJ> iterator;

    map<KEY,OBJ*>& _map;

    associative_container_keyval(map<KEY,OBJ*>& __map):
      _map(__map){}

    int size() const{
      return _map.size();
    }

    iterator begin() const{
      return iterator(_map.begin());
    }

    iterator end() const{
      return iterator(_map.end());
    }

  };


  template<typename KEY,typename OBJ>
  class associative_container{
  public:

    typedef associative_container_val_iterator<KEY,OBJ> viterator;
    typedef associative_container_keyval<KEY,OBJ> _keyval;

    //vector<OBJ*> v;
    mutable map<KEY,OBJ*> _map;
    
    ~associative_container(){
      for(auto p: _map) 
	delete p.second;
    }

    associative_container(){}


  public: // ---- Copying ------------------------------------------------------------------------------------
    

    associative_container(const associative_container& x){
      for(auto p: x._map)
	_map[p.first]=new OBJ(*p.second);
    }

    associative_container(associative_container&& x){
      _map=x._map;
      x._map.clear();
    }

    associative_container& operator=(const associative_container& x){
      clear();
      for(auto p: x._map)
	_map[p.first]=new OBJ(*p.second);
      return *this;
    }

    associative_container& operator=(associative_container&& x){
      clear();
      _map=x._map;
      x._map.clear();
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return _map.size();
    }

    OBJ* pointer_to(const KEY& key){
      if(!exists(key)) insert(key,new OBJ());
      return _map[key];
    }

    OBJ& operator[](const KEY& key){
      if(!exists(key)) insert(key,new OBJ());
      return *_map[key];
    }

    const OBJ& operator[](const KEY& key) const{
      assert(exists(key));
      return *_map[key];
    }

    OBJ& first(){
      return *_map.begin()->second;
    }

    const OBJ& first() const{
      return *_map.begin()->second;
    }

    void insert(const KEY& key, OBJ* obj){
      assert(!exists(key));
      _map[key]=obj;
    }

    void insert(const KEY& key, const OBJ& obj){
      assert(!exists(key));
      _map[key]=new OBJ(obj);
    }

    void insert(const KEY& key, OBJ&& obj){
      assert(!exists(key));
      _map[key]=new OBJ(std::move(obj)); 
    }

    bool exists(const KEY& key) const{
      return _map.find(key)!=_map.end();
    }

    viterator begin() const{
      return viterator(_map.begin());
    }

    viterator end() const{
      return viterator(_map.end());
    }

    _keyval keyval() const{
      return _keyval(_map);
    }

    void clear(){
      for(auto p: _map) 
	delete _map.second;
      _map.clear();
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:_map){
	oss<<indent<<p.first.str()<<":"<<endl;
	oss<<p.second->str(indent+"  ")<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const associative_container& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif

