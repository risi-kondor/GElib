/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2022, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _labeled_tree
#define _labeled_tree

#include "Cnine_base.hpp"
#include "int_tree.hpp"


namespace cnine{

  template<typename TYPE>
  class labeled_tree{
  public:

    TYPE label;
    vector<labeled_tree*> children;

    ~labeled_tree(){
      for(auto& p:children)
	delete p;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    labeled_tree(const TYPE& x): 
      label(x){}

    labeled_tree(TYPE&& x): 
      label(std::move(x)){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    labeled_tree(const labeled_tree& x):
      label(x.label){
      for(auto& p: x.children)
	children.push_back(new labeled_tree(*p));
    }

    labeled_tree(labeled_tree&& x):
      label(std::move(x.label)),
      children(std::move(x.children)){
      x.children.clear();
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    int_tree as_int_tree() const{
      int_tree r;

      int m=children.size();
      auto root=r.add_root(label,m);
      for(int i=0; i<m; i++){
	auto child=root.add_child(i,children[i]->label,children[i]->children.size());
	children[i]->as_int_tree(child,i);
      }
      return r;
    }

    void as_int_tree(int_tree::node& node, const int i) const{
      int m=children.size();
      for(int i=0; i<m; i++){
	auto child=node.add_child(i,children[i]->label,children[i]->children.size());
	children[i]->as_int_tree(child,i);
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    labeled_tree* branchp(const TYPE& x) const{
      for(auto p:children)
	if(p->label==x) return p;
      return nullptr;
    }

    labeled_tree& branch(const TYPE& x){
      for(auto p:children)
	if(p->label==x) return *p;
      auto r=new labeled_tree(x);
      children.push_back(r);
      return *r;
    }

    void push_back(labeled_tree* x){
      children.push_back(x);
    }

    void add_rooted_path(const vector<TYPE> x){
      if(x.size()==0) return;
      branch(x[0]).add_rooted_path(vector<TYPE>(x.begin()+1,x.end()));
    }


  public: // ---- Contains -----------------------------------------------------------------------------------


    bool contains(const TYPE& x) const{
      return branchp(x)!=nullptr;
    }

    bool contains(const vector<TYPE>& x) const{
      if(x.size()==0) return true;
      auto p=branchp(x[0]);
      if(!p) return false;
      return p->contains(vector<TYPE>(x.begin()+1,x.end()));
    }

    bool contains_rooted_path_consisting_of(const std::vector<TYPE>& x) const{
      set<TYPE> s(x.begin(),x.end());
      return contains_rooted_path_consisting_of(s);
    }

    bool contains_rooted_path_consisting_of(const std::set<TYPE>& x) const{
      //cout<<"set:"; for(auto p:x) cout<<p; cout<<endl;
      auto it=x.find(label);
      if(it==x.end()) return false;
      if(x.size()==1) return true;

      set<TYPE> setd(x);
      setd.erase(label);
      for(auto& p:children){
	//cout<<"child"<<endl;
	if(p->contains_rooted_path_consisting_of(setd)) return true;
      }

      return false;
    }


  public: // ---- traversals ---------------------------------------------------------------------------------


    vector<TYPE> depth_first_traversal() const{
      vector<TYPE> R;
      depth_first([&](const TYPE& x){R.push_back(x);});
      return R;
    }

    vector<pair<TYPE,int> > indexed_depth_first_traversal() const{
      vector<pair<TYPE,int> > R;
      indexed_depth_first([&](const TYPE& x, const int ix){
	  R.push_back(pair<TYPE,int>(x,ix));},0);
      return R;
    }


  public: // ---- for_each -----------------------------------------------------------------------------------


    void for_each_maximal_path(const std::function<void(const vector<TYPE>&)> lambda) const{
      vector<TYPE> prefix;
      for_each_maximal_path(prefix,lambda);
    }

    void for_each_maximal_path(vector<TYPE> prefix, const std::function<void(const vector<TYPE>&)> lambda) const{
      prefix.push_back(label);
      if(children.size()==0) lambda(prefix);
      else
	for(auto& p: children)
	  p->for_each_maximal_path(prefix,lambda);
    }

    void depth_first(const std::function<void(const TYPE&)> lambda) const{
      lambda(label);
      for(auto& p: children){
	p->depth_first(lambda);
      }
    }

    int indexed_depth_first(const std::function<void(const TYPE&, const int)> lambda, const int ix, const int old=-1) const{
      lambda(label,old);
      int j=1;
      for(auto& p: children){
	j+=p->indexed_depth_first(lambda,ix+j,ix);
      }
      return j;
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

    friend ostream& operator<<(ostream& stream, const labeled_tree& x){
      stream<<x.str(); return stream;}

  };


}

#endif 
    /*
    bool contains_rooted_path_consisting_of(const std::set<TYPE>& x) const{
      if(x.size()==0) return true;
      cout<<"set:"; for(auto p:x) cout<<p; cout<<endl;
      for(auto& p:children){
	cout<<"testing "<<p->label<<endl;
	auto it=x.find(p->label);
	if(it==x.end()) continue;
	set<TYPE> setd(x);
	setd.erase(p->label);
	if(p->contains_rooted_path_consisting_of(setd)) return true;
	cout<<"failed"<<endl;
      }
      return false;
    }
    */

