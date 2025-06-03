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


#ifndef _CnineSparseTensor
#define _CnineSparseTensor

#include "Cnine_base.hpp"
#include "map_of_maps.hpp"
#include "Gdims.hpp"
#include "Gindex.hpp"
#include "map_of_lists.hpp"


namespace cnine{

  template<typename TYPE>
  class SparseTensor{
  public:

    unordered_map<Gindex,TYPE> map;
    map_of_maps<int,int,TYPE> map2; 

    Gdims dims;
    bool mom=false;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SparseTensor(const Gdims& _dims):
      dims(_dims){
      if(dims.size()==2) mom=true;
    }

    SparseTensor(const Gdims& _dims, const map_of_lists<int,int>& mask, const fill_sequential& fill):
      dims(_dims){
      CNINE_ASSRT(dims.size()==2);
      mom=true;
      int t=0; mask.for_each([&](const int i, const int j){set(i,j,t++);});
    }


  public: // ---- Named constructors -------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


    int nfilled() const{
      if(mom) return map2.nfilled();
      return map.size();
    }

    bool is_filled(const int i0) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      return (map.find({i0})!=map.end());
    }

    bool is_filled(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      if(mom) return map2.is_filled(i0,i1);
      else return (map.find({i0,i1})!=map.end());
    }

    bool is_filled(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      return (map.find({i0,i1,i2})!=map.end());
    }

    bool is_filled(const Gindex& ix) const{
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      if(mom) return map2.is_filled(ix[0],ix[1]);
      else return (map.find(ix)!=map.end());
    }


    // ---- operator() const


    TYPE operator()(const int i0) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      auto it=map.find(i0);
      if(it==map.end()) return TYPE();
      return it->second;
    }

    TYPE operator()(const int i0, const int i1) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      if(mom) return map2(i0,i1);
      else{
	auto it=map.find({i0,i1});
	if(it==map.end()) return TYPE();
	return it->second;
      }
    }

    TYPE operator()(const int i0, const int i1, const int i2) const{
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      auto it=map.find({i0,i1,i2});
      if(it==map.end()) return TYPE();
      return it->second;
    }

   TYPE operator()(const Gindex& ix) const{
     //CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      auto it=map.find(ix);
      if(it==map.end()) return TYPE();
      return it->second;
    }


    // ---- operator() 


    TYPE& operator()(const int i0){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      auto it=map.find(i0);
      if(it==map.end()) return TYPE();
      return it->second;
    }

    TYPE& operator()(const int i0, const int i1){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      if(mom) return map2(i0,i1);
      else{
	auto it=map.find({i0,i1});
	if(it==map.end()) return TYPE();
	return it->second;
      }
    }

    TYPE& operator()(const int i0, const int i1, const int i2) {
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      auto it=map.find({i0,i1,i2});
      if(it==map.end()) return TYPE();
      return it->second;
    }

   TYPE operator()(const Gindex& ix){
     //CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      auto it=map.find(ix);
      if(it==map.end()) return TYPE();
      return it->second;
    }


    // ---- set


    void set(const int i0, const TYPE& x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,string(__PRETTY_FUNCTION__)));
      map[{i0}]=x;
    }

    void set(const int i0, const int i1, const TYPE& x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,string(__PRETTY_FUNCTION__)));
      if(mom) map2.set(i0,i1,x);
      else map[{i0,i1}]=x;
    }

    void set(const int i0, const int i1, const int i2, const TYPE& x){
      CNINE_CHECK_RANGE(dims.check_in_range(i0,i1,i2,string(__PRETTY_FUNCTION__)));
      map[{i0,i1,i2}]=x;
    }

    void set(const Gindex& ix, const TYPE& x){
      CNINE_CHECK_RANGE(dims.check_in_range(ix,string(__PRETTY_FUNCTION__)));
      if(mom) map2.set(ix[0],ix[1],x);
      else map[ix]=x;
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_nonzero(const std::function<void(const Gindex&, const TYPE&)>& lambda) const{
      if(mom) map2.for_each([&](const int& i, const int& j, const TYPE& x){
	  lambda(Gindex(i,j),x);});
      else{
	for(auto& p:map)
	  lambda(p.first,p.second);
      }
    }

    void for_each_nonzero(const std::function<void(const Gindex&, TYPE&)>& lambda){
      if(mom) map2.for_each([&](const int& i, const int& j, const TYPE& x){
	  lambda(Gindex(i,j),x);});
      else{
	for(auto& p:map)
	  lambda(p.first,p.second);
      }
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
     for_each_nonzero([&](const Gindex& ix, const TYPE& x){
	 oss<<indent<<ix<<"->"<<x<<endl;});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SparseTensor& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
