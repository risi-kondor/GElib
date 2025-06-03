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

#ifndef _cnine_double_indexed_map
#define _cnine_double_indexed_map

#include "Cnine_base.hpp"
#include <map>
#include "map_of_maps.hpp"


namespace cnine{

  template<typename KEY1, typename KEY2, typename TYPE>
  class double_indexed_map{
  public:

    mutable map<KEY1,std::map<KEY2,TYPE> > rmap;
    mutable map<KEY1,std::map<KEY2,TYPE> > cmap;


  public: // ---- Constructors ------------------------------------------------------------------------------


    double_indexed_map(){}

    double_indexed_map(const map_of_maps<KEY1,KEY2,TYPE>& x){
      for(auto& p:x){
	map<KEY2,TYPE> a;
	for(auto& q:p.second)
	  a[q.first]=q.second;
	rmap[p.first]=a;
      }
      make_cmap();
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    bool is_empty() const{
      for(auto q:rmap)
	if(q.second->size()>0)
	  return false;
      return true;
    }

    int size() const{
      int t=0;
      for(auto& p:rmap)
	t+=p.second.size();
      return t;
    }

    int nfilled() const{
      int t=0;
      for(auto& p:rmap)
	t+=p.second.size();
      return t;
    }

    bool is_filled(const KEY1& i, const KEY2& j) const{
      auto it=rmap.find(i);
      if(it==rmap.end()) return false;
      auto it2=it->second.find(j);
      if(it2==it->second.end()) return false;
      return true;
    }

    TYPE operator()(const KEY1& i, const KEY2& j) const{
      auto it=rmap.find(i);
      if(it==rmap.end()) return TYPE();
      auto it2=it->second.find(j);
      if(it2==it->second.end()) return TYPE();
      return it2->second;
    }

    void set(const KEY1& i, const KEY2& j, const TYPE& x){
      rmap[i][j]=x;
      cmap[j][i]=x;
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each(const std::function<void(const KEY1&, const KEY2&, const TYPE&)>& lambda) const{
      for(auto& p:rmap)
	for(auto& q:p.second)
	  lambda(p.first,q.first,q.second);
    }

    void for_each_in_row(const KEY1& x, const std::function<void(const KEY2&, const TYPE&)>& lambda) const{
      for(auto& p: rmap[x])
	lambda(p.first,p.second);
    }

    void for_each_in_columns(const KEY2& x, const std::function<void(const KEY1&, const TYPE&)>& lambda) const{
      if(cmap.size()==0) make_cmap();
      for(auto& p: cmap[x])
	lambda(p.first,p.second);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    void make_cmap() const{
      cmap.clear();
      for(auto& p: rmap)
	for(auto& q:p.second)
	  cmap[p.first][q.first]=p.second;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    string classname() const{
      return "double_indexed_map";
    }

    string repr() const{
      return "double_indexed_map";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p: rmap){
	oss<<indent<<p.first<<":(";
	for(auto& q: p.second)
	  oss<<q.first<<":"<<q.second<<",";
	oss<<"\b)"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const double_indexed_map& v){
      stream<<v.str(); return stream;}

  };


}

#endif 
