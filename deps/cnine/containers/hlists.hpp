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

#ifndef _cnine_hlists
#define _cnine_hlists

#include "Cnine_base.hpp"
#include "array_pool.hpp"
#include "map_of_lists.hpp"


namespace cnine{

  template<typename TYPE>
  class hlists: public array_pool<TYPE>{
  public:

    typedef array_pool<TYPE> BASE;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dir;
    using BASE::tail;
    using BASE::size;


  public: // ---- Constructors -------------------------------------------------------------------------------


    hlists(const vector<TYPE>& heads, const vector<int>& lengths):
      BASE(lengths.size(),std::accumulate(lengths.begin(),lengths.end(),0)+lengths.size(),fill_reserve()){
      int N=size();
      CNINE_ASSRT(heads.size()==N);
      for(int i=0; i<N; i++){
	dir.set(i,0,tail);
	dir.set(i,1,lengths[i]+1); // changed
	arr[tail]=heads[i];
	tail+=lengths[i]+1;
      }
    }

    hlists(const vector<TYPE>& heads, const vector<int>& lengths, const fill_noalloc& dummy):
      BASE(lengths.size(),std::accumulate(lengths.begin(),lengths.end(),0)+lengths.size(),fill_reserve()){
      int N=size();
      CNINE_ASSRT(heads.size()==N);
      for(int i=0; i<N; i++){
	dir.set(i,0,tail);
	dir.set(i,1,1); // changed
	arr[tail]=heads[i];
	tail+=lengths[i]+1;
      }
    }

    hlists(const map_of_lists<TYPE,TYPE>& map):
      BASE(map.size(),map.size()+map.tsize(),fill_reserve()){
      int i=0;
      for(auto& p:map){
	int m=p.second.size();
	dir.set(i,0,tail);
	dir.set(i,1,m+1);
	arr[tail]=p.first;
	for(int j=0; j<m; j++)
	  arr[tail+j+1]=p.second[j];
	tail+=m+1;
	i++;
      }
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    hlists(const hlists& x):
      BASE(x){}

    hlists(hlists&& x):
      BASE(std::move(x)){}

    hlists& operator=(const hlists& x){
      BASE::operator=(x);
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------
    

    int total() const{
      return BASE::total()-size();
    }

    int size_of(const int i) const{
      CNINE_ASSRT(i<size());
      return dir(i,1)-1;
    }

    int head(const int i) const{
      CNINE_ASSRT(i<size());
      return arr[dir(i,0)];
    }

    void set_head(const int i, const TYPE v){
      CNINE_ASSRT(i<size());
      arr[dir(i,0)]=v;
    }

    int& ref(const int i, const int j) const{
      CNINE_ASSRT(i<size());
      CNINE_ASSRT(j<dir(i,1)-1);
      return arr[dir(i,0)+j+1];
    }

    int operator()(const int i, const int j) const{
      CNINE_ASSRT(i<size());
      CNINE_ASSRT(j<dir(i,1)-1);
      return arr[dir(i,0)+j+1];
    }

    void set(const int i, const int j, const TYPE v){
      CNINE_ASSRT(i<size());
      CNINE_ASSRT(j<dir(i,1)-1);
      arr[dir(i,0)+j+1]=v;
    }

    vector<TYPE> operator()(const int i) const{
      CNINE_ASSRT(i<size());
      int addr=dir(i,0)+1;
      int len=dir(i,1)-1;
      vector<TYPE> R(len);
      for(int i=0; i<len; i++)
	R[i]=arr[addr+i];
      return R;
    }


  public: // ---- lambdas ------------------------------------------------------------------------------------


    void for_each(const std::function<void(const TYPE, const vector<TYPE>&)>& lambda) const{
      int n=size();
      for(int i=0; i<n; i++)
	lambda(head(i),(*this)(i));
    }

    void for_each(const std::function<void(const TYPE, const TYPE)>& lambda) const{
      int n=size();
      for(int i=0; i<n; i++){
	TYPE h=head(i);
	int offs=dir(i,0)+1;
	int n=dir(i,1)-1;
	for(int j=0; j<n; j++)
	  lambda(h,arr[offs+j]);
      }
    }

    void for_each_of(const int i, std::function<void(const TYPE)> lambda) const{
      CNINE_ASSRT(i<size());
      int offs=dir(i,0)+1;
      int n=dir(i,1)-1;
      for(int j=0; j<n; j++)
	lambda(arr[offs+j]);
    }

    void for_each_of(const int i, std::function<void(const TYPE&)>& lambda) const{
      CNINE_ASSRT(i<size());
      int offs=dir(i,0)+1;
      int n=dir(i,1)-1;
      for(int j=0; j<n; j++)
	lambda(arr[offs+j]);
    }


  public: // ---- push_back ----------------------------------------------------------------------------------


    void push_back(const int len){
      BASE::push_back(len+1);
    }

    void push_back(const TYPE x, const vector<TYPE>& v){
      int len=v.size()+1;
      if(tail+len>BASE::memsize)
	BASE::reserve(std::max(2*BASE::memsize,tail+len));
      arr[tail]=x;
      for(int i=0; i<len-1; i++)
	arr[tail+i+1]=v[i];
      dir.push_back(tail,len);
      tail+=len;
    }

    void push_back(const TYPE h, const std::set<TYPE>& x){
      int len=x.size()+1;
      if(tail+len>BASE::memsize)
	BASE::reserve(std::max(2*BASE::memsize,tail+len));
      arr[tail]=h;
      int i=0; 
      for(auto p:x)
	arr[tail+i+1]=p;
      dir.push_back(tail,len);
      tail+=len;
    }

    void push_back(const int h, const initializer_list<TYPE>& x){
      push_back(h,vector<TYPE>(x));
    }

    void push_back(const int i, const TYPE v){
      BASE::push_back(i,v);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "hlists";
    }

    string repr() const{
      return "hlists";
    }

    string str(const string indent="") const{
      //for(int i=0; i<tail; i++) cout<<arr[i]<<" "; cout<<endl;
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<head(i)<<":(";
	for(int j=0; j<size_of(i); j++)
	  oss<<(*this)(i,j)<<",";
	if(size_of(i)>0) oss<<"\b";
	oss<<")\n";
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const hlists& v){
      stream<<v.str(); return stream;}

  };



}

#endif 
