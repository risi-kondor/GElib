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

#ifndef _CnineMapOfLists
#define _CnineMapOfLists

#include "Cnine_base.hpp"
#include "IntTensor.hpp"


namespace cnine{



  template<typename KEY, typename ITEM>
  class map_of_lists: public unordered_map<KEY,std::vector<ITEM> >{
  public:

    typedef unordered_map<KEY,std::vector<ITEM> > BASE;
    

    using BASE::find;
    using BASE::begin;
    using BASE::end;
    using BASE::operator[];

    //unordered_map<KEY,std::vector<ITEM> > data;


  public: // ---- Constructors -------------------------------------------------------------------------------


    map_of_lists(){}

    map_of_lists(const initializer_list<pair<KEY,initializer_list<ITEM> > >& list){
      for(auto& p:list){
	KEY key=p.first;
	for(auto& q:p.second)
	  push_back(key,q);
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int total() const{
      int t=0;
      for(auto& p:*this)
	t+=p.second.size();
      return t;
    }

    int tsize() const{
      int t=0;
      for(auto& p:*this)
	t+=p.second.size();
      return t;
    }

    size_t max_size() const{
      size_t t=0;
      for(auto& p:*this)
	bump(t,p.second.size());
      return t;
    }

    void push_back(const KEY& x, const ITEM& y){
      auto it=find(x);
      if(it==end()) (*this)[x]=vector<ITEM>({y});
      else it->second.push_back(y);
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------

  
    void for_each(const std::function<void(const KEY&, const ITEM&)>& lambda) const{
      for(auto& p:*this)
	for(auto& q: p.second)
	  lambda(p.first,q);
    }

    void for_each_in_list(const KEY& x, const std::function<void(const ITEM&)>& lambda) const{
      auto it=find(x);
      if(it==end()) return;
      auto& v=it->second;
      for(int i=0; i<v.size(); i++)
	lambda(v[i]);
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "map_of_lists";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:*this){
	oss<<indent<<p.first<<": (";
	for(int i=0; i<p.second.size()-1; i++)
	  oss<<p.second[i]<<",";
	if(p.second.size()>0) 
	  oss<<p.second.back();
	oss<<")"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const map_of_lists& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif 
  // old version to scrap
  /*
  template<typename KEY, typename ITEM>
  class map_of_lists{
  public:


    unordered_map<KEY,std::vector<ITEM> > data;


  public: // ---- Constructors -------------------------------------------------------------------------------


    map_of_lists(){}

    map_of_lists(const initializer_list<pair<KEY,initializer_list<ITEM> > >& list){
      for(auto& p:list){
	KEY key=p.first;
	for(auto& q:p.second)
	  push_back(key,q);
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int total() const{
      int t=0;
      for(auto& p:data)
	t+=p.second.size();
      return t;
    }

    void push_back(const KEY& x, const ITEM& y){
      auto it=data.find(x);
      if(it==data.end()) data[x]=vector<ITEM>({y});
      else it->second.push_back(y);
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------

  
    void for_each(const std::function<void(const KEY&, const ITEM&)>& lambda) const{
      for(auto& p:data)
	for(auto& q: p.second)
	  lambda(p.first,q);
    }

    void for_each_in_list(const KEY& x, const std::function<void(const ITEM&)>& lambda) const{
      auto it=data.find(x);
      if(it==data.end()) return;
      auto& v=it->second;
      for(int i=0; i<v.size(); i++)
	lambda(v[i]);
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "map_of_lists";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:data){
	oss<<indent<<p.first<<": (";
	for(int i=0; i<p.second.size()-1; i++)
	  oss<<p.second[i]<<",";
	if(p.second.size()>0) 
	  oss<<p.second.back();
	oss<<")"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const map_of_lists& x){
      stream<<x.str(); return stream;
    }

  };
  */

