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


#ifndef _CnineGindex
#define _CnineGindex

#include "Cnine_base.hpp"
#include "Gdims.hpp"


namespace cnine{
    

  class Gindex: public vector<int>{
  public:

    typedef std::size_t size_t;


    Gindex(){}

    Gindex(const int k, const fill_zero& dummy): 
      vector<int>(k,0){}

    Gindex(const int k, const fill_raw& dummy): 
      vector<int>(k){}

    Gindex(const fill_zero& dummy){
    }

    Gindex(const vector<int>& x){
      for(auto p:x) if(p>=0) push_back(p);
    }

    Gindex(const initializer_list<int>& list): vector<int>(list){}

    Gindex(const int i0):
      Gindex({i0}){}

    Gindex(const int i0, const int i1): vector<int>(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }

    Gindex(const int i0, const int i1, const int i2): vector<int>(3){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
    }

    Gindex(const int i0, const int i1, const int i2, const int i3): vector<int>(4){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
    }

    Gindex(const int i0, const int i1, const int i2, const int i3, const int i4): vector<int>(5){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
      (*this)[4]=i4;
    }

    Gindex(size_t a, const Gdims& dims): 
      vector<int>(dims.size()){
      for(int i=size()-1; i>=0; i--){
	(*this)[i]=a%dims[i];
	a=a/dims[i];
      }
    }

    Gindex(size_t a, vector<int> strides): 
      vector<int>(strides.size()){
      for(int i=size()-1; i>=1; i--){
	(*this)[i]=(a%strides[i-1])/strides[i];
      }
      (*this)[0]=a/strides[0];
    }

    

  public:
    
    int k() const{
      return size();
    }

    size_t asize() const{
      int t=1;
      for(int i=0; i<size(); i++)
	t*=(*this)[i];
      return t;
    }

    int operator()(const int i) const{
      return (*this)[i];
    }

    void set(const int i, const int x){
      (*this)[i]=x;
    }

    size_t operator()(const vector<int>& strides) const{
      assert(strides.size()>=size());
      size_t t=0; 
      for(int i=0; i<size(); i++) 
	t+=(*this)[i]*strides[i];
      return t;
    }

    size_t operator()(const Gdims& dims) const{
      assert(dims.size()>=size());
      int s=1;
      size_t t=0; 
      for(int i=size()-1; i>=0; i--){
	t+=(*this)[i]*s;
	s*=dims[i];
      }
      return t;
    }

    size_t to_int(const Gdims& dims) const{
      assert(dims.size()>=size());
      int s=1;
      size_t t=0; 
      for(int i=size()-1; i>=0; i--){
	t+=(*this)[i]*s;
	s*=dims[i];
      }
      return t;
    }

    Gindex cat(const Gindex& y) const{
      Gindex R(size()+y.size(),fill_raw());
      for(int i=0; i<size(); i++) R[i]=(*this)[i];
      for(int i=0; i<y.size(); i++) R[size()+i]=y[i];
      return R;
    }

    Gindex chunk(int beg, int n=-1) const{
      if(beg<0) beg=size()+beg;
      if(n==-1) n=size()-beg;
      Gindex R(n,fill_raw());
      for(int i=0; i<n; i++)
	R[i]=(*this)[beg+i];
      return R;
    }

    Gindex operator+(const Gdims& y) const{
      CNINE_ASSRT(y.size()==size());
      Gindex R(*this);
      for(int i=0; i<size(); i++)
	R[i]+=y[i];
      return R;
    }
      
    bool operator==(const Gindex& x) const{
      if(size()!=x.size()) return false;
      for(int i=0; i<size(); i++)
	if((*this)[i]!=x[i]) return false;
      return true;
    }

    bool operator<=(const vector<int>& x) const{
      if(size()!=x.size()) return false;
      for(int i=0; i<size(); i++)
	if((*this)[i]>x[i]) return false;
      return true;
    }


  public:

    void check_range(const Gdims& dims) const{
      if(size()!=dims.size()) throw std::out_of_range("index "+str()+" out of range of dimensions "+dims.str());
      for(int i=0; i<size(); i++)
	if((*this)[i]<0) throw std::out_of_range("index "+str()+" out of range of dimensions "+dims.str());
      for(int i=0; i<size(); i++)
	if((*this)[i]>=dims[i]) throw std::out_of_range("index "+str()+" out of range of dimensions "+dims.str());
    }

    void check_arange(const Gdims& dims) const{
      if(size()!=dims.size()) throw std::out_of_range("index "+str()+" out of range of cell array dimensions "+dims.str());
      for(int i=0; i<size(); i++)
	if((*this)[i]<0) throw std::out_of_range("index "+str()+" out of range of cell array dimensions "+dims.str());
      for(int i=0; i<size(); i++)
	if((*this)[i]>=dims[i]) throw std::out_of_range("index "+str()+" out of range of cell array dimensions "+dims.str());
    }


  public:

    string str() const{
      string str="("; 
      int k=size();
      for(int i=0; i<k; i++){
	str+=std::to_string((*this)[i]);
	if(i<k-1) str+=",";
      }
      return str+")";
    }

    string str_bare() const{
      string str; 
      int k=size();
      for(int i=0; i<k; i++){
	str+=std::to_string((*this)[i]);
	if(i<k-1) str+=",";
      }
      return str;
    }

    string repr() const{
      return "<cnine::Gdims"+str()+">";
    }


  friend ostream& operator<<(ostream& stream, const Gindex& v){
    stream<<v.str(); return stream;}

  };



}


namespace std{
  template<>
  struct hash<cnine::Gindex>{
  public:
    size_t operator()(const cnine::Gindex& ix) const{
      size_t t=hash<int>()(ix[0]);
      for(int i=1; i<ix.size(); i++) t=(t<<1)^hash<int>()(ix[i]);
      return t;
    }
  };
}




#endif

