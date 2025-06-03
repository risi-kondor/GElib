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


#ifndef _CninePermutation
#define _CninePermutation

#include "Cnine_base.hpp"


namespace cnine{


  class permutation{
  public:

  public:

    int n;
    int* p;

    ~permutation(){delete[] p;}


  public: // Constructors

    permutation():
      permutation(1){};

    permutation(const int _n): 
      n(_n){
      p=new int[n];
    }

    permutation(const int _n, const cnine::fill_raw& dummy): 
      n(_n){
      p=new int[n];
    }

    permutation(const int _n, const cnine::fill_identity& dummy): 
      n(_n){
      p=new int[n];
      for(int i=0; i<n; i++) p[i]=i;
    }

    permutation(const initializer_list<int> list): 
      permutation(list.size(), cnine::fill_raw()){
      int i=0; for(auto v:list) p[i++]=v;
      if(!is_valid()){cerr<<"Invalid permutation"<<endl;}
    }	

    permutation(const vector<int> x): 
      n(x.size()){
      p=new int[n]; 
      for(int i=0; i<n; i++) p[i]=x[i];
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    permutation(const permutation& x): n(x.n){
      p=new int[n]; 
      for(int i=0; i<n; i++) p[i]=x.p[i];
    }

    permutation(permutation&& x): n(x.n) {
      p=x.p; x.p=nullptr;
    }
  
    permutation operator=(const permutation& x){
      n=x.n; delete[] p; p=new int[n]; 
      for(int i=0; i<n; i++) p[i]=x.p[i];
      return *this;
    }

    permutation& operator=(permutation&& x){
      if (this!=&x) {n=x.n; delete[] p; p=x.p; x.p = NULL;}
      return *this;
    }


  public: // ---- Named Constructors ------------------------------------------------------------------------


    static permutation identity(const int _n){
      permutation p(_n,cnine::fill_raw()); 
      for(int i=0; i<_n; i++) p.p[i]=i; 
      return p;
    }

    static permutation Identity(const int _n){
      permutation p(_n,cnine::fill_raw()); 
      for(int i=0; i<_n; i++) p.p[i]=i; 
      return p;
    }

    static permutation transposition(const int _n, const int i, const int j){
      permutation p(_n,cnine::fill_raw()); 
      for(int i=0; i<_n; i++) p.p[i]=i; 
      p.p[i]=j;
      p.p[j]=i;
      return p;
    }
      
    static permutation contiguous_cycle(const int _n, const int a, const int b){
      permutation p(_n,cnine::fill_raw());
      if(a<b){
	for(int i=1; i<a; i++) p.p[i]=i; 
	for(int i=b+1; i<=_n; i++) p.p[i]=i;
	for(int i=a; i<=b-1; i++) p.p[i]=i+1;
	p.p[b-1]=a;
      }else{
	for(int i=1; i<b; i++) p.p[i]=i; 
	for(int i=a+1; i<=_n; i++) p.p[i]=i;
	for(int i=b+1; i<=a; i++) p.p[i]=i-1;
	p.p[b-1]=a;
      }
      return p;
    }

    static permutation random(const int n){
      permutation p(n,cnine::fill_raw());

      vector<int> v(n);
      for(int i=0; i<n; i++) v[i]=i;

      for(int i=0; i<n; i++){
	std::uniform_int_distribution<> distr(0,n-1-i);
	int t=distr(rndGen);
	p.p[i]=v[t];
	v.erase(v.begin()+t);
      }

      return p;
    }

    template<typename TYPE>
    static permutation ordering(const vector<TYPE> v){
      int n=v.size();
      permutation r(n);
      vector<bool> done(n,false);
      for(int i=0; i<n; i++){
	int least=-1;
	for(int j=0; j<n; j++)
	  if(!done[j] && (least==-1 || v[j]<v[least])) least=j;
	done[least]=true;
	r[i]=least;
      }
      return r;
    }

    static permutation ordering(const int n, const std::function<bool(const int, const int)>& less){
      permutation r(n);
      vector<bool> done(n,false);
      for(int i=0; i<n; i++){
	int least=-1;
	for(int j=0; j<n; j++)
	  if(!done[j] && (least==-1 || less(j,least))) least=j;
	done[least]=true;
	r[i]=least;
      }
      return r;
    }


  public: // ---- Conversions -------------------------------------------------------------------------------

    
  public: // ---- Access ------------------------------------------------------------------------------------


    int getn() const{
      return n;
    }

    int size() const{
      return n;
    }

    int operator()(const int i) const{
      return p[i];
    }

    int& operator[](const int i){
      return p[i];
    }

    void set(const int i, const int j){
      p[i]=j;
    }

    bool operator==(const permutation& x) const{
      if(n!=x.n) return false;
      for(int i=0; i<n; i++) if(p[i]!=x.p[i]) return false;
      return true;
    }

    permutation operator!() const{
      permutation result(n,cnine::fill_raw());
      for(int i=0; i<n; i++) result.p[p[i]]=i;
      return result;
    }


  public: // ---- Operations --------------------------------------------------------------------------------


    permutation operator*(const permutation& x) const{
      permutation result(n,cnine::fill_raw());
      for(int i=0; i<n; i++) result.p[i]=p[x(i)];
      return result;
    }

    permutation& operator*=(const permutation& x){
      for(int i=0; i<n; i++) p[i]=x(p[i]);
      return *this;
    }

    permutation inverse() const{
      permutation r(n,cnine::fill_raw());
      for(int i=0; i<n; i++)
	r.p[p[i]]=i;
      return r;
    }

    permutation inv() const{
      return inverse();
    }

    bool is_valid() const{
      vector<bool> a(n,false);
      for(int i=0; i<n; i++){
	if(p[i]<0 || p[i]>n-1 || a[p[i]]) return false;
	a[p[i]]=true;
      }
      return true;
    }

    template<typename TYPE>
    vector<TYPE> reorder(const vector<TYPE>&  x){
      int n=x.size();
      vector<TYPE> r(n);
      for(int i=0; i<n; i++)
	r[i]=x[(*this)[i]];
      return r;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<"[ "; 
      for(int i=0; i<n; i++) oss<<p[i]<<" ";
      oss<<"]"; 
      return oss.str();
    }
     
    
    friend ostream& operator<<(ostream& stream, const permutation& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif
