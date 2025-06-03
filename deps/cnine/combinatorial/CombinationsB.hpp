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

#ifndef _CombinationsB
#define _CombinationsB

#include "Cnine_base.hpp"


namespace cnine{


  class Combination: public vector<int>{
  public:

    Combination(const int k):
      vector<int>(k){}

    Combination(const initializer_list<int>& x): 
      vector<int>(x){}


  public:

    Combination& shift(const int j){
      for(int i=0; i<size(); i++)
	(*this)[i]+=j;
      return *this;
    }

    Combination prepend(const int j) const{
      Combination R(size()+1);
      R[0]=j;
      for(int i=0; i<size(); i++)
	R[i+1]=(*this)[i];
      return R;
    }

    Combination rest() const{
      Combination R(size()-1);
      for(int i=0; i<size()-1; i++)
	R[i]=(*this)[i+1];
      return R;
    }

    

  public:

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"[ ";
      for(int i=0; i<size(); i++) 
	oss<<(*this)[i]<<" ";
      oss<<"]";
      return oss.str();
    }      

  };




  class CombinationsB{
  public:

    int n;
    int m;
    int N;
    vector<CombinationsB*> sub;

    CombinationsB(const int _n, const int _m): n(_n), m(_m){
    }


  public:

    int getN() const{
      return N;
    }


    Combination operator[](const int i) const{
      assert(i<N);
      if(m==1) return Combination({i});
	
      int j=0;
      int t=0;
      while(i>=t+sub[j]->N){
	t+=sub[j]->N;
	j++;
      }
      return (*sub[j])[i-t].shift(j+1).prepend(j);
    }


    int index(const Combination& v) const{
      if(v.size()==1) return v[0];
      int t=0;
      for(int i=0; i<v[0]; i++)
	t+=sub[i]->N;
      return t+sub[v[0]]->index(v.rest().shift(-(v[0]+1)));
    }


    void for_each(std::function<void(Combination) > fn) const{

      if(m==1){
	for(int i=0; i<n; i++)
	  fn(Combination({i}));
	return;
      }

      for(int i=0; i<n-m+1; i++)
	sub[i]->for_each([fn,i](Combination _x){
	    Combination x(_x);
	    fn(x.shift(i+1).prepend(i));
	  });
    }



  };


}

#endif 
