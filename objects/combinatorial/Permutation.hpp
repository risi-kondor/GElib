
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _Permutation
#define _Permutation

//#include "CyclicShifts.hpp"

#include "GElib_base.hpp"

namespace GElib{


  class Permutation{
  public:

  public:

    int n;
    int* p;


  public: // Constructors


    Permutation(const int _n, const cnine::fill_raw& dummy): 
      n(_n){
      p=new int[n];
    }

    Permutation(const int _n, const cnine::fill_identity& dummy): 
      n(_n){
      p=new int[n];
      for(int i=0; i<n; i++) p[i]=i+1;
    }

    Permutation(const initializer_list<int> list): 
      Permutation(list.size(), cnine::fill_raw()){
      int i=0; for(auto v:list) p[i++]=v;
    }	

    Permutation(const vector<int> x): n(x.size()){
      p=new int[n]; 
      for(int i=0; i<n; i++) p[i]=x[i];
    }

    /*
      Permutation(const CyclicShifts& x): n(x.size()){
      p=new int[n]; for(int i=0; i<n; i++) p[i]=i+1;
      for(int i=0; i<n; i++){
      for(int j=0; j<i; j++) if (p[j]>=x[i]) p[j]++;
      p[i]=x[i];
      }
      }

      Permutation(const CyclicShifts& x, const int _n): n(_n){
      p=new int[n]; for(int i=0; i<n; i++) p[i]=i+1;
      for(int i=0; i<x.size(); i++){
      int ii=n-x.size()+i;
      for(int j=0; j<ii; j++) if (p[j]>=x[i]) p[j]++;
      p[ii]=x[i];
      }
      }
    */



    Permutation(const Permutation& x): n(x.n){
      p=new int[n]; 
      for(int i=0; i<n; i++) p[i]=x.p[i];
    }

    Permutation(Permutation&& x): n(x.n) {
      p=x.p; x.p = NULL;
    }
  
    Permutation operator=(const Permutation& x){
      n=x.n; delete[] p; p=new int[n]; 
      for(int i=0; i<n; i++) p[i]=x.p[i];
      return *this;
    }

    Permutation& operator=(Permutation&& x){
      if (this!=&x) {n=x.n; delete[] p; p=x.p; x.p = NULL;}
      return *this;
    }

    ~Permutation(){delete[] p;}


  public: // named constructors

    static Permutation Identity(const int _n){
      Permutation p(_n,cnine::fill_raw()); 
      for(int i=0; i<_n; i++) p.p[i]=i+1; 
      return p;
    }

    //static Permutation Random(const int n);

  
  public: // Access

    int operator()(const int i) const{
      return p[i-1];
    }

    int operator[](const int i) const{
      return p[i];
    }

    int& operator[](const int i){
      return p[i];
    }

    bool operator==(const Permutation& x) const{
      if(n!=x.n) return false;
      for(int i=0; i<n; i++) if(p[i]!=x.p[i]) return false;
      return true;
    }

    Permutation operator!() const{
      Permutation result(n,cnine::fill_raw());
      for(int i=0; i<n; i++) result.p[p[i]-1]=i+1;
      return result;
    }


  public: 

    Permutation operator*(const Permutation& x) const{
      Permutation result(n,cnine::fill_raw());
      for(int i=0; i<n; i++) result.p[i]=p[x(i+1)-1];
      return result;
    }

    Permutation& operator*=(const Permutation& x){
      for(int i=0; i<n; i++) p[i]=x(p[i]);
      return *this;
    }

  public: // I/O 

    string str(const string indent="") const{
      ostringstream oss;
      oss<<"[ "; 
      for(int i=0; i<n; i++) oss<<p[i]<<" ";
      oss<<"]"; 
      return oss.str();
    }
     
    
    friend ostream& operator<<(ostream& stream, const Permutation& x){
      stream<<x.str(); return stream;
    }

  };

//ostream& operator<<(ostream& stream, const Permutation& sigma);
//LatexMath& operator<<(LatexMath& stream, const Permutation& sigma);

}


#endif
