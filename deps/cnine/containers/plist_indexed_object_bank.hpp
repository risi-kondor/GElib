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


#ifndef _plist_indexed_object_bank
#define _plist_indexed_object_bank

#include "Cnine_base.hpp"
#include "observable.hpp"


namespace cnine{

  template<typename PTR>
  class plist: public vector<PTR>{
  public:
    
    using vector<PTR>::size;

    plist(){}
    plist(const vector<PTR>& x):
      vector<PTR>(x){}

    bool operator==(const plist& x) const{
      if(size()!=x.size()) return false;
      for(int i=0; i<size(); i++)
	if((*this)[i]!=x[i]) return false;
      return true;
    }
  };



  template<typename KEY, typename OBJ>
  class plist_indexed_object_bank: public unordered_map<plist<KEY*>,OBJ>{
  public:

    using unordered_map<plist<KEY*>,OBJ>::size;
    using unordered_map<plist<KEY*>,OBJ>::insert;
    using unordered_map<plist<KEY*>,OBJ>::find;
    using unordered_map<plist<KEY*>,OBJ>::erase;
    using unordered_map<plist<KEY*>,OBJ>::end;


    std::function<OBJ(const vector<KEY*>&)> make_obj;

    unordered_map<KEY*,vector<plist<KEY*> > > memberships;
    observer<KEY> observers;
  
    ~plist_indexed_object_bank(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    plist_indexed_object_bank():
      make_obj([](const vector<KEY*>& x){cout<<"empty object in bank"<<endl; return OBJ();}),
      observers([this](KEY* p){erase_all_involving(p);}){}

    plist_indexed_object_bank(std::function<OBJ(const vector<KEY*>&)> _make_obj):
      make_obj(_make_obj),
      observers([this](KEY* p){erase_all_involving(p);}){}

  
  public: // ---- Private ------------------------------------------------------------------------------------


    void erase_all_involving(KEY* p){
      auto it=memberships.find(p);
      CNINE_ASSRT(it!=memberships.end());
      for(auto& q:it->second)
	erase(q);
      memberships.erase(it);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    //OBJ operator()(KEY& key){
    //return (*this)(&key);
    //}

    //OBJ operator()(const KEY& key){
    //return (*this)(&const_cast<KEY&>(key));
    //}

    OBJ& operator()(const plist<KEY*>&  key){
      auto it=find(key);
      if(it!=end()) 
	return it->second;
      //cout<<"not in cache"<<endl;

      for(auto p:key){
	observers.add(p);
	memberships[p].push_back(key);
      }
      
      auto q=insert({key,make_obj(key)});
      return q.first->second;
    }

  };


}


namespace std{
  template<typename PTR>
  struct hash<cnine::plist<PTR> >{
  public:
    size_t operator()(const cnine::plist<PTR>& x) const{
      size_t r=1;
      for(auto p:x)
	r=(r<<1)^hash<PTR>()(p);
      return r;
    }
  };
}


#endif 


  /*
  template<typename KEY, typename OBJ>
  class shared_object_bank: public unordered_map<KEY*,shared_ptr<OBJ> >{
  public:

    using unordered_map<KEY*,shared_ptr<OBJ> >::find;
    using unordered_map<KEY*,shared_ptr<OBJ> >::erase;


    std::function<OBJ*(const KEY&)> make_obj;
    observer<KEY> observer;
    
    ~shared_object_bank(){
    }


  public: // ---- Constructors --------------------------------------------------------------------------------


    shared_object_bank():
      make_obj([](const KEY& x){cout<<"shared_obj_bank error"<<endl; return nullptr;}),
      observer([this](KEY* p){erase(p);}){}

    shared_object_bank(std::function<OBJ*(const KEY&)> _make_obj):
      make_obj(_make_obj),
      observer([this](KEY* p){erase(p);}){}


  public: // ---- Access -------------------------------------------------------------------------------------


    shared_ptr<OBJ> operator()(KEY& key){
      return (*this)(&key);
    }

    shared_ptr<OBJ> operator()(const KEY& key){
      return (*this)(&const_cast<KEY&>(key));
    }

    shared_ptr<OBJ> operator()(KEY* keyp){
      auto it=find(keyp);
      if(it!=unordered_map<KEY*, shared_ptr<OBJ> >::end()) 
	return it->second;

      auto new_obj=shared_ptr<OBJ>(make_obj(*keyp));
      (*this)[keyp]=new_obj;
      observer.add(keyp);
      return new_obj;
    }

  };
  */
