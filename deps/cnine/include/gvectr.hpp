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


#ifndef __CnineGvec
#define __CnineGvec

#include "Cnine_base.hpp"


namespace cnine{


  template<typename TYPE, typename SUB>
  class Gvec: public vector<TYPE>{
  public:

    typedef vector<TYPE> BASE;

    using BASE::BASE;
    using BASE::size;
    using BASE::begin;
    using BASE::end;

    Gvec(){}

    Gvec(const vector<TYPE>& x):
      BASE(x){}


  public: // ---- Merging ------------------------------------------------------------------------------------

    
    Gvec(const Gvec& x, const Gvec& y):
      Gvec(x.size()+y.size()){
      std::copy(x.begin(),x.end(),begin());
      std::copy(y.begin(),y.end(),begin()+x.size());
    }

    Gvec(const Gvec& x, const TYPE v, const Gvec& y):
      Gvec(x.size()+y.size()+1){
      std::copy(x.begin(),x.end(),begin());
      std::copy(y.begin(),y.end(),begin()+x.size()+1);
      (*this)[x.size()]=v;
    }

    Gvec(const TYPE b, const Gvec& x, const Gvec& y):
      Gvec(x.size()+y.size()+1){
      (*this)[0]=b;
      std::copy(x.begin(),x.end(),begin()+1);
      std::copy(y.begin(),y.end(),begin()+x.size()+1);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    template<typename TYPE2>
    vector<TYPE2> to_vector() const{
      vector<TYPE2> R(size());
      for(int i=0; i<size(); i++)
	R[i]=(*this)[i];
      return R;
    }
    

  public: // ---- Access -------------------------------------------------------------------------------------

    
    TYPE operator()(const int i) const{
      if(i<0) return (*this)[size()+i];
      return (*this)[i];
    }

    TYPE back(const int i=0) const{
      return (*this)[size()-1-i];
    }

    Gvec& set(const int i, const TYPE x){
      (*this)[i]=x;
      return *this;
    }

    Gvec& set_back(const TYPE x){
      (*this)[size()-1]=x;
      return *this;
    }

    Gvec& set_back(const int i, const TYPE x){
      (*this)[size()-1-i]=x;
      return *this;
    }

    TYPE first() const{
      return (*this)[0];
    }

    TYPE last() const{
      return (*this)[size()-1];
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    TYPE max() const{
      return *std::max_element(begin(),end());
    }

    TYPE min() const{
      return *std::min_element(begin(),end());
    }

    SUB operator+(const Gvec& y) const{
      CNINE_ASSRT(y.size()==size());
      Gvec R(*this);
      for(int i=0; i<size(); i++)
	R[i]+=y[i];
      return R;
    }

    SUB insert(const int j, const TYPE x) const{
      Gvec R(size()+1);
      std::copy(begin(),begin()+j,R.begin());
      if(j<size()) std::copy(begin()+j,end(),R.begin()+j+1);
      R[j]=x;
      return R;
    }

    SUB insert(const int j, const vector<TYPE>& x) const{
      Gvec R(size()+x.size());
      std::copy(begin(),begin()+j,R.begin());
      std::copy(x.begin(),x.end(),R.begin()+j);
      std::copy(begin()+j,end(),R.begin()+j+x.size());
      return R;
    }

    SUB insert(const int j, int n, const TYPE x) const{
      Gvec R(size()+n);
      std::copy(begin(),begin()+j,R.begin());
      std::copy(begin()+j,end(),R.begin()+j+n);
      for(int i=0; i<n; i++)
	R[j+i]=x;
      return R;
    }

    SUB remove(const int j) const{
      CNINE_ASSRT(j<size());
      Gvec R(size()-1);
      if(j>0) std::copy(begin(),begin()+j,R.begin());
      if(j<size()-1) std::copy(begin()+j+1,end(),R.begin()+j);
      return R;
    }

    SUB remove(const vector<int>& v) const{
      return cnine::except(*this,v);
    }

    SUB replace(const int j, const TYPE x) const{
      CNINE_ASSRT(j<size());
      Gvec R(*this);
      R[j]=x;
      return R;
    }

    SUB prepend(const TYPE x) const{
      Gvec R(size()+1);
      R[0]=x;
      std::copy(begin(),end(),R.begin()+1);
      return R;
    }

    SUB append(const TYPE x) const{
      Gvec R(size()+1);
      std::copy(begin(),end(),R.begin());
      R[size()]=x;
      return R;
    }

    SUB cat(const Gvec& y) const{
      Gvec R(size()+y.size());
      std::copy(begin(),end(),R.begin());
      std::copy(y.begin(),y.end(),R.begin()+size());
      return R;
    }

    SUB chunk(int beg, int n=-1) const{
      if(beg<0) beg=size()+beg;
      if(n==-1) n=size()-beg;
      Gvec R(n);
      std::copy(begin()+beg,begin()+beg+n,R.begin());
      return R;
    }

    SUB permute(const vector<int>& p) const{
      CNINE_ASSRT(p.size()==size());
      Gvec R;
      R.resize(size());
      for(int i=0; i<p.size(); i++)
	R[i]=(*this)[p[i]];
      return R;
    }




  public: // ---- I/O ----------------------------------------------------------------------------------------


    static string classname(){
      return "Gvec";
    }

    string repr() const{
      return "<cnine::Gvec"+str()+">";
    }

    string str() const{
      ostringstream oss;
      int k=size();
      oss<<"(";
      for(int i=0; i<k; i++){
	oss<<(*this)[i];
	if(i<k-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Gvec& x){
      stream<<x.str(); return stream;
    }

  };

  
  template<typename TYPE, typename SUB> 
  inline bool operator<(const Gvec<TYPE,SUB>& x, const Gvec<TYPE,SUB>& y){
    int n=std::min(x.size(),y.size());
    int i=0;
    while(i<n && x[i]==y[i]) {i++;}
    if(i==n) return x.size()<y.size();
    return x[i]<y[i];
  }

  template<typename TYPE, typename SUB> 
  inline bool operator<=(const Gvec<TYPE,SUB>& x, const Gvec<TYPE,SUB>& y){
    int n=std::min(x.size(),y.size());
    int i=0;
    while(i<n && x[i]==y[i]) {i++;}
    if(i==n) return x.size()<=y.size();
    return x[i]<=y[i];
  }

}

#endif 


    //Gdims(const BASE& x):
    //BASE(x){}

    //Gdims(const initializer_list<int>& x):
    //BASE(x){}

    //Gdims(const int n):
    //BASE(n){}
    /*
    bool operator<=(const Gvec& x) const{
      if(size()>x.size()) return false;
      for(size_t i=0; i<size(); i++)
	if((*this)[i]>x[i]) return false;
      return true;
    }
    */
