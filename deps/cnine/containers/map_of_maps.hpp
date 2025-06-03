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

#ifndef _CnineMapOfMaps
#define _CnineMapOfMaps

#include "Cnine_base.hpp"
#include <unordered_map>
#include "TensorView.hpp"
#include "int_pool.hpp"


namespace cnine{

  template<typename KEY1, typename KEY2, typename TYPE>
  class map_of_maps{
  public:

    mutable unordered_map<KEY1,std::unordered_map<KEY2,TYPE> > data;

   


  public: // ---- Conversions --------------------------------------------------------------------------------


    int_pool as_int_pool(){
      int_pool r(data.size(),size());
      for(auto& p: data){
	int start=r.tail();
	int i=0;
	for(auto q:p)
	  r.arr[start+(i++)]=q.first;
	r.add_vec(i);
      }
      return r;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_empty() const{
      for(auto q:data)
	if(q.second->size()>0)
	  return false;
      return true;
    }

    int size() const{
      int t=0;
      for(auto& p:data)
	t+=p.second.size();
      return t;
    }

    int nfilled() const{
      int t=0;
      for(auto& p:data)
	t+=p.second.size();
      return t;
    }

    bool is_filled(const KEY1& i, const KEY2& j) const{
      auto it=data.find(i);
      if(it==data.end()) return false;
      auto it2=it->second.find(j);
      if(it2==it->second.end()) return false;
      return true;
    }

    TYPE operator()(const KEY1& i, const KEY2& j) const{
      auto it=data.find(i);
      if(it==data.end()) return TYPE();
      auto it2=it->second.find(j);
      if(it2==it->second.end()) return TYPE();
      return it2->second;
    }

    //TYPE& operator()(const KEY1& i, const KEY2& j){
    //return data[i][j];
    //}

    void set(const KEY1& i, const KEY2& j, const TYPE& x){
      data[i][j]=x;
    }

    bool operator==(const map_of_maps<KEY1,KEY2,TYPE>& x) const{ 
      if(!subset_of(x)) return false;
      if(!x.subset_of(*this)) return false;
      return true;
      //for(auto& p:data)
      //for(auto q: p.second)
      //if(x(p.first,q.first)!=q.second) return false;
      //for(auto& p:x.data)
      //for(auto q: p.second)
      //if((*this)(p.first,q.first)!=q.second) return false;
      return true;
    }

    bool subset_of(const map_of_maps<KEY1,KEY2,TYPE>& x) const{
      for(auto& p:data){
	auto it1=x.data.find(p.first);
	if(it1==x.data.end()) return false;
	for(auto& q: p.second){
	  auto it2=it1->second.find(q.first);
	  if(it2==it1->second.end()) return false;
	  if(it2->second!=q.second) return false;
	}
      }
      return true;
    }
      

  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each(const std::function<void(const KEY1&, const KEY2&, const TYPE&)>& lambda) const{
      for(auto& p:data)
	for(auto& q:p.second)
	  lambda(p.first,q.first,q.second);
    }

    //void for_each(const KEY1& x, const std::function<void(const KEY2&, const TYPE&)>& lambda) const{
    //for(auto& p:data)
    //for(auto& q:p.second)
    //  lambda(p.first,q.first,q.second);
    //}



  public: // ---- I/O ---------------------------------------------------------------------------------------


    
  };

}


namespace std{

  template<typename KEY1, typename KEY2, typename TYPE>
  struct hash<cnine::map_of_maps<KEY1,KEY2,TYPE> >{
  public:
    size_t operator()(const cnine::map_of_maps<KEY1,KEY2,TYPE>& x) const{
      size_t t=1;
      for(auto& p:x.data)
	for(auto& q:p.second){
	  t=(t^hash<KEY1>()(p.first))<<1;
	  t=(t^hash<KEY2>()(q.first))<<1;
	  t=(t^hash<TYPE>()(q.second))<<1;
	}
      return t;
    }
  };
}

#endif 
