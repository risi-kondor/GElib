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


#ifndef __GstridesB
#define __GstridesB

#include "Cnine_base.hpp"
#include "Gdims.hpp"

namespace cnine{


  class GstridesB: public vector<int>{
  public:

    int offset=0;
    bool regular=false;

    GstridesB(){}

    GstridesB(const int k, const fill_raw& dummy): 
      vector<int>(k){}

    GstridesB(const initializer_list<int>& lst):
      vector<int>(lst){}

    GstridesB(const int i0): vector<int>(1){
      (*this)[0]=i0;
    }

    GstridesB(const int i0, const int i1): vector<int>(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }

    GstridesB(const int i0, const int i1, const int i2): vector<int>(3){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
    }

    GstridesB(const int i0, const int i1, const int i2, const int i3): vector<int>(4){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
    }

    GstridesB(const int i0, const int i1, const int i2, const int i3, const int i4): vector<int>(5){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
      (*this)[4]=i4;
    }


    GstridesB(const Gdims& dims, const int s0=1): 
      vector<int>(dims.size()){
      int k=dims.size();
      assert(k>0);
      (*this)[k-1]=s0;
      for(int i=k-2; i>=0; i--)
      (*this)[i]=(*this)[i+1]*dims[i+1];
      regular=true;
    }

    GstridesB(const vector<int>& x):
      vector<int>(x){
    }


  public:

    int operator()(const int i) const{
      if(i<0) return (*this)[size()+i];
      return (*this)[i];
    }

    int back(const int i=0) const{
      return (*this)[size()-1-i];
    }

    bool is_regular(const Gdims& dims) const{
      CNINE_ASSRT(size()==dims.size());
      int k=size();
      int t=1;
      for(int i=k-1; i>=0; i--){
	if((*this)[i]!=t) return false;
	t*=dims[i];
      }
      return true;
    }

    int memsize(const Gdims& dims) const{
      CNINE_ASSRT(size()==dims.size());
      int t=0;
      for(int i=0; i<size(); i++)
	t=std::max(t,(*this)[i]*dims[i]);
      return t;
    }

    int operator()(const vector<int>& ix) const{
      CNINE_ASSRT(ix.size()<=size());
      int t=offset;
      for(int i=0; i<ix.size(); i++)
	t+=(*this)[i]*ix[i];
      return t;
    }

    int offs(const int i0) const{
      return offset+i0*(*this)[0];
    }

    int offs(const int i0, const int i1) const{
      return offset+i0*(*this)[0]+i1*(*this)[1];
    }

    int offs(const int i0, const int i1, const int i2) const{
      return offset+i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2];
    }

    int offs(const int i0, const int i1, const int i2, const int i3) const{
      return offset+i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2]+i3*(*this)[3];
    }

    int combine(const vector<int>& v) const{
      int t=0;
      for(auto p:v){
	assert(p<size());
	t+=(*this)[p];
      }
      return t;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    GstridesB fuse(const int a, const int n) const{
      GstridesB R(size()-n+1,fill_raw());
      for(int i=0; i<a; i++) R[i]=(*this)[i];
      for(int i=0; i<size()-(a+n-1); i++) R[a+i]=(*this)[a+n+i-1];
      return R;
    }
    
    GstridesB remove(const int j) const{
      GstridesB R;
      assert(j<size());
      if(size()==1){
	R.push_back(1);
	return R;
      }
      if(j<0){
	for(int i=0; i<size(); i++)
	  if(i!=size()+j) R.push_back((*this)[i]);
      }else{
	for(int i=0; i<size(); i++)
	  if(i!=j) R.push_back((*this)[i]);
      }
      return R;
    }
    
    GstridesB append(const int s) const{
      GstridesB R(*this);
      R.push_back(s);
      return R;
    }

    GstridesB chunk(const int beg, int n=-1) const{
      if(n==-1) n=size()-beg;
      GstridesB R(n,fill_raw());
      for(int i=0; i<n; i++)
	R[i]=(*this)[beg+i];
      return R;
    }

    int offs(int j, const GstridesB& source) const{
      assert(source.size()==size());
      int t=0;
      for(int i=0; i<size(); i++){
	int r=j/source[i];
	t+=(*this)[i]*r;
	j-=r*source[i];
      }
      return t;
    }

    GstridesB& inc_offset(const int i){
      offset+=i;
      return *this;
    }

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      int k=size();
      oss<<indent<<"[";
      for(int i=0; i<k; i++){
	oss<<(*this)[i];
	if(i<k-1) oss<<",";
      }
      oss<<"]";
      return oss.str();
    }

  };

}


#endif 
