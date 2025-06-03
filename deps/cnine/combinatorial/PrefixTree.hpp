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

#ifndef _PrefixTree
#define _PrefixTree

#include "Cnine_base.hpp"
#include <unordered_map>


namespace cnine{

  template<typename TYPE>
  class PrefixTree{
  public:

    unordered_map<TYPE,PrefixTree*> children;

    ~PrefixTree(){
      for(auto& p:children)
	delete p.second;
    }


    PrefixTree(){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    PrefixTree(const PrefixTree& x){
      for(auto& p: x.children)
	children[p.first]=new PrefixTree(*p.second);
    }

    PrefixTree(PrefixTree&& x):
      children(std::move(x.children)){
      x.children.clear();
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    bool find(const TYPE& x) const{
      return branch(x)!=nullptr;
    }

    bool find(const vector<TYPE>& x) const{
      if(x.size()==0) return true;
      auto p=branch(x[0]);
      if(!p) return false;
      return p->find(vector<TYPE>(x.begin()+1,x.end()));
    }

    bool contains_some_permutation_of(std::vector<TYPE>& x) const{
      set<TYPE> s(x.begin(),x.end());
      contains_some_permutation_of(s);
    }

    /*
    bool contains_some_permutation_of(std::set<TYPE>& x) const{ // faulty!
      if(x.size()==0) return true;
      for(auto& p:children){
	auto it=x.find(p.first);
	if(it==x.end()) return false;
	set<TYPE> setd(set);
	setd.erase(p.first);
	return p.second->contains_some_permutation_of(setd);
      }
      return false;
    }
    */

    PrefixTree* branch(const TYPE& x) const{
      auto it=children.find(x);
      if(it!=children.end()) return it->second;
      else return nullptr;
    }

    PrefixTree& get_branch(const TYPE& x){
      auto it=children.find(x);
      if(it!=children.end()) return *(it->second);
      children[x]=new PrefixTree();
      return *children[x];
    }

    void add_path(const vector<TYPE> x){
      if(x.size()==0) return;
      get_branch(x[0]).add_path(vector<TYPE>(x.begin()+1,x.end()));
    }

    void for_each_maximal_path(const std::function<void(const vector<TYPE>&)> lambda) const{
      vector<TYPE> prefix;
      for_each_maximal_path(prefix,lambda);
    }

    void for_each_maximal_path(vector<TYPE>& prefix, const std::function<void(const vector<TYPE>&)> lambda) const{
      if(children.size()==0) lambda(prefix);
      for(auto& p: children){
	prefix.push_back(p.first);
	p.second->for_each_maximal_path(prefix,lambda);
	prefix.pop_back();
      }
    }

    void depth_first(const std::function<void(const TYPE&)> lambda) const{
      for(auto& p: children){
	lambda(p.first);
	p.second->depth_first(lambda);
      }
    }

    int depth_first(const std::function<void(const TYPE&, const int)> lambda, const int ix) const{
      int j=0;
      for(auto& p: children){
	lambda(p.first,ix);
	j+=p.second->depth_first(lambda,ix+j+1)+1;
      }
      return j;
    }

    vector<TYPE> depth_first_traversal() const{
      vector<TYPE> R;
      depth_first([&](const TYPE& x){R.push_back(x);});
      return R;
    }

    vector<TYPE> indexed_depth_first_traversal() const{
      vector<pair<TYPE,int> > R;
      depth_first([&](const TYPE& x, const int ix){
	  R.push_back(pair<TYPE,int>(x,ix));
	},-1);
      return R;
    }

    vector<TYPE> indexed_depth_first_traversal(const TYPE& root_label) const{
      vector<pair<TYPE,int> > R;
      R.push_back(pair<TYPE,int>(root_label,-1));
      depth_first([&](const TYPE& x, const int ix){
	  R.push_back(pair<TYPE,int>(x,ix));
	},0);
      return R;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for_each_maximal_path([&](const vector<TYPE>& x){
	  oss<<indent<<"(";
	  for(int i=0; i<x.size()-1; i++) oss<<x[i]<<",";
	  if(x.size()>0) oss<<x.back();
	  oss<<")"<<endl;
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const PrefixTree& x){
      stream<<x.str(); return stream;}

  };

    

}

#endif 
