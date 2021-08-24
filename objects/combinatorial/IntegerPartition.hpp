
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _IntegerPartition
#define _IntegerPartition


namespace GElib{

  class IntegerPartition{
  public: 

    int k;
    int* p;

    
  public:

    IntegerPartition(): k(0){
      p=nullptr;
    }

    IntegerPartition(const int _k, const cnine::fill_raw& dummy): k(_k){
      p=new int[k];
    }

    IntegerPartition(const int n, const cnine::fill_identity& dummy): k(1){
      p=new int[1]; 
      p[0]=n;
    }

    IntegerPartition(const int _k, const cnine::fill_zero& dummy): k(_k){
      p=new int[k]; 
      for(int i=0; i<k; i++) p[i]=0;
    }

    IntegerPartition(const initializer_list<int> list): 
      IntegerPartition(list.size(),cnine::fill_raw()){
      int i=0; for(auto v:list) p[i++]=v;
    }	


  public: // copying 

    IntegerPartition(const IntegerPartition& x):k(x.k){
      p=new int[k]; 
      for(int i=0; i<k; i++) p[i]=x.p[i];
    }

    IntegerPartition& operator=(const IntegerPartition& x){
      k=x.k; delete[] p; p=new int[k]; 
      for(int i=0; i<k; i++) p[i]=x.p[i]; 
      return *this;
    }
  
    IntegerPartition(IntegerPartition&& x): k(x.k) {
      p=x.p; x.p=nullptr;}
  
    IntegerPartition& operator=(IntegerPartition&& x){
      if (this!=&x) {k=x.k; delete[] p; p=x.p; x.p=nullptr;}
      return *this;
    }
  
    ~IntegerPartition(){delete[] p;}


  private:

    int factorial(int x) const{
      return x==0 ? 1 : x*=factorial(x-1);
    }


  public: // Access

    int height() const{
      return k;
    }

    int getn() const{
      int n=0; 
      for(int i=0; i<k; i++) n+=p[i]; 
      return n;
    }

    int& operator[](const int r){
      return p[r];
    }

    int operator[](const int r) const{
      return p[r];
    }

    int& operator()(const int r){
      return p[r-1];
    }

    int operator()(const int r) const{
      return p[r-1];
    }


  public:

    int hooklength() const {
      int res = factorial(getn());
      for(int r=1; r<=k; r++){
	for(int c=1; c<=p[r-1]; c++){
	  int right = p[r-1] - c;
	  int below = 0;
	  for(int i=r+1; i<=k; i++)
	    below += (p[i-1] >=c ? 1 : 0);
	  res /= (right+below+1);
	}
      }
      return res;
    };

    bool extendable(const int i) const{
      if(i==k) return true;
      if(i==0) return true;
      if(p[i-1]>p[i]) return true;
      return false;
    }

    bool shortenable(const int i) const{
      if(i==k-1) return true;
      if(p[i+1]<p[i]) return true;
      return false;
    }

    IntegerPartition& add(const int r, const int m=1){
      if(r<=k){p[r]+=m; return *this;}
      int* newp=new int[k+1]; 
      for(int i=0; i<k; i++) newp[i]=p[i]; newp[k]=m;
      k++; delete[] p; p=newp; 
      return *this; 
    }

    IntegerPartition& remove(const int r, const int m=1){
      if(p[r]>m){p[r]-=m; return *this;}
      int* newp=new int[k-1]; 
      for(int i=0; i<k-1; i++) newp[i]=p[i]; 
      k--; delete[] p; p=newp; 
      return* this; 
    }

    bool operator==(const IntegerPartition& x) const{
      int i=0; for(; i<k && i<x.k; i++) if (p[i]!=x.p[i]) return false;
      for(;i<k; i++) if (p[i]!=0) return false;
      for(;i<x.k; i++) if(x.p[i]!=0) return false;
      return true;
    }


  public: // I/O

    string str(const string indent="") const{
      ostringstream oss;
      oss<<"[ "; 
      for(int i=0; i<k; i++) oss<<p[i]<<" ";
      oss<<"]"; 
      return oss.str();
    }
     
    
    friend ostream& operator<<(ostream& stream, const IntegerPartition& x){
      stream<<x.str(); return stream;
    }


  };

}

namespace std{
  template<>
  struct hash<GElib::IntegerPartition>{
  public:
    size_t operator()(const GElib::IntegerPartition& x) const{
      size_t r=1;
      for(int i=0; i<x.k; i++) r=(r<<1)^hash<int>()(x.p[i]);
      return r;
    }
  };
}


#endif
