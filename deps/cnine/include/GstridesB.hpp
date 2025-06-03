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

#include <climits>
#include "Cnine_base.hpp"
#include "gvectr.hpp"
#include "Gdims.hpp"
#include "Gstrides.hpp"

namespace cnine{


  class TensorPackDir;


  class GstridesB: public Gvec<std::size_t,GstridesB>{
  public:

    typedef Gvec<std::size_t,GstridesB> BASE;
    typedef std::size_t size_t;
    friend class TensorPackDir;

    using BASE::operator[];

    using BASE::operator();
    using BASE::insert;
    using BASE::remove;
    using BASE::replace;
    using BASE::prepend;
    using BASE::append;
    using BASE::cat;
    using BASE::chunk;
    using BASE::permute;


    GstridesB(){}

    GstridesB(const vector<size_t>& x):
      BASE(x){}

    GstridesB(const initializer_list<size_t>& x):
      BASE(x){}

    GstridesB(const initializer_list<int>& lst){
      for(auto p:lst)
	push_back(p);
    }

    GstridesB(const int k, const fill_raw& dummy): 
      BASE(k){}

    //GstridesB(const vector<int>& x):
    //BASE(x.size()){
    //for(int i=0; i<x.size(); i++)
    //(*this)[i]=x[i];
    //}

    GstridesB copy() const{
      return GstridesB(*this);
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static GstridesB raw(const int k){
      return BASE(k);
    }

    static GstridesB zero(const int k){
      return BASE(k,0);
    }


  public: // ---- Merging -------------------------------------------------------------------------


    GstridesB(const GstridesB& d1, const size_t v, const GstridesB& d2):
      BASE(d1,v,d2){}

    GstridesB(const size_t b, const GstridesB& d1, const GstridesB& d2):
      BASE(b,d1,d2){}


  public: // ---- Constructing from Gdims -------------------------------------------------------------------


    GstridesB(const Gdims& dims, const int s0=1): 
      BASE(dims.size()){
      int k=dims.size();
      assert(k>0);
      (*this)[k-1]=s0;
      for(int i=k-2; i>=0; i--)
      (*this)[i]=(*this)[i+1]*dims[i+1];
    }


  public: // ---- ATEN --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN
    GstridesB(const at::Tensor& T):
      GstridesB(T.dim(),fill_raw()){
      for(int i=0; i<size() ; i++)
	(*this)[i]=T.stride(i);
    }
#endif 


  public: // ---- Access ------------------------------------------------------------------------------------


    GstridesB& set(const int i, const int x){
      BASE::set(i,x);
      return *this;
    }

    GstridesB& set_back(const int x){
      BASE::set_back(x);
      return *this;
    }

    GstridesB& set_back(const int i, const int x){
      BASE::set_back(i,x);
      return *this;
    }

    size_t memsize(const Gdims& dims) const{
      CNINE_ASSRT(size()==dims.size());
      if(dims.asize()==0) return 0;
      size_t t=0;
      for(int i=0; i<size(); i++)
	t=std::max(t,(*this)[i]*dims[i]);
      return t;
    }

    GstridesB transp() const{
      int len=size();
      CNINE_ASSRT(len>=2);
      if(len==2) return GstridesB({(*this)[1],(*this)[0]});
      GstridesB r(*this);
      std::swap(r[len-2],r[len-1]);
      return r;
    }

    Gstrides reals() const{
      Gstrides R(size(),fill_raw());
      for(int i=0; i<size(); i++)
	R[i]=(*this)[i]*2;
      return R;
    }


  public: // ---- Indexing -----------------------------------------------------------------------------------


    size_t operator()(const vector<int>& ix) const{
      CNINE_ASSRT(ix.size()<=size());
      size_t t=0; //offset;
      for(int i=0; i<ix.size(); i++)
	t+=(*this)[i]*ix[i];
      return t;
    }

    size_t offs(const vector<int>& ix) const{
      CNINE_ASSRT(ix.size()<=size());
      size_t t=0;
      for(int i=0; i<ix.size(); i++)
	t+=(*this)[i]*ix[i];
      return t;
    }

    size_t offs(const int i, const vector<int>& ix) const{
      CNINE_ASSRT(ix.size()<=size()-1);
      size_t t=((*this)[0]);
      for(int i=0; i<ix.size(); i++)
	t+=(*this)[i+1]*ix[i];
      return t;
    }

    size_t offs(const int i0) const{
      return i0*(*this)[0];
    }

    size_t offs(const int i0, const int i1) const{
      return i0*(*this)[0]+i1*(*this)[1];
    }

    size_t offs(const int i0, const int i1, const int i2) const{
      return i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2];
    }

    size_t offs(const int i0, const int i1, const int i2, const int i3) const{
      return i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2]+i3*(*this)[3];
    }

    size_t offs(const int i0, const int i1, const int i2, const int i3, const int i4) const{
      return i0*(*this)[0]+i1*(*this)[1]+i2*(*this)[2]+i3*(*this)[3]+i4*(*this)[4];
    }

    size_t combine(const vector<int>& v) const{
      size_t t=0;
      for(auto p:v){
	assert(p<size());
	t+=(*this)[p];
      }
      return t;
    }


  public: // ---- Regularity ---------------------------------------------------------------------------------


    bool is_decreasing() const{
      if(size()==0) return true;
      for(int i=1; i<size(); i++)
	if((*this)[i]>(*this)[i-1]) return false;
      return true;
    }
    
    bool is_regular(const Gdims& dims) const{
      CNINE_ASSRT(size()==dims.size());
      size_t k=size();
      int t=1;
      for(int i=k-1; i>=0; i--){
	if((*this)[i]!=t) return false;
	t*=dims[i];
      }
      return true;
    }

    bool is_contiguous(const Gdims& dims) const{
      CNINE_ASSRT(size()==dims.size());
      if(is_regular(dims)) return true;

      BASE v(*this);
      int nz=0; 
      for(int i=0; i<size(); i++) 
	if(v[i]>0) nz++;

      size_t t=1;
      for(int i=0; i<nz; i++){
	auto it=std::find(v.begin(),v.end(),t);
	if(it==v.end()) return false;
	//int a=it-v.begin();
	*it=0;
	t*=dims[it-v.begin()];
      }

      return true;
    }

    GstridesB map(const GindexMap& map) const{
      CNINE_ASSRT(map.ndims()==size());
      int n=map.size();
      GstridesB R=GstridesB::zero(n);
      for(int i=0; i<n; i++){
	auto& ix=map[i];
	size_t t=0;
	for(int j=0; j<ix.size(); j++){
	  CNINE_ASSRT(ix[j]<size());
	  t+=(*this)[ix[j]];
	}
	R[i]=t;
      }
      return R;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    vector<int> ordering() const{
      if(size()==0) return vector<int>();
      int n=size();
      vector<size_t> v(*this);
      vector<int> r(n);
      for(int i=0; i<n; i++){
	auto j=std::min_element(v.begin(),v.end());
	r[i]=j-v.begin();
	*j=INT_MAX;
      }
      return r;
    }

    vector<int> descending_ordering() const{
      if(size()==0) return vector<int>();
      int n=size();
      vector<size_t> v(*this);
      vector<int> r(n);
      for(int i=0; i<n; i++){
	auto j=std::max_element(v.begin(),v.end());
	r[i]=j-v.begin();
	*j=0;
      }
      return r;
    }

    size_t offs(int j, const GstridesB& source) const{
      assert(source.size()==size());
      size_t t=0;
      for(int i=0; i<size(); i++){
	size_t r=j/source[i];
	t+=(*this)[i]*r;
	j-=r*source[i];
      }
      return t;
    }


  public: // ---- Fusing ------------------------------------------------------------------------------------


    bool fusible(const Gdims& dims, const vector<int>& ordering){
      int n=size();
      CNINE_ASSRT(dims.size()==n);
      CNINE_ASSRT(ordering.size()==n);
      if(n==0) return true;

      CNINE_ASSRT(ordering[0]<n);
      for(int i=0; i<n; i++){
	CNINE_ASSRT(ordering[i]<n);
	if((*this)[ordering[i]]!=(*this)[ordering[i-1]]*dims[ordering[i-1]]) return false;
      }
      return true;
    }

     GstridesB fuse(const int a, const int n) const{
      GstridesB R(size()-n+1,fill_raw());
      for(int i=0; i<a; i++) R[i]=(*this)[i];
      for(int i=0; i<size()-(a+n-1); i++) R[a+i]=(*this)[a+n+i-1];
      return R;
    }
    
    pair<size_t,int> fuser(const Gdims& dims) const{
      auto p=ordering();
      int n=size();
      if(n==-1) return make_pair((size_t)0,-1);
      int mins=(*this)[p[0]];
      for(int i=0; i<n-1; i++)
	if((*this)[p[i+1]]!=(*this)[p[i]]*dims[p[i]])
	  return make_pair((size_t)0,-1);
      return std::pair<size_t,int>((*this)[p[0]],(*this)[p[n-1]]*dims[p[n-1]]/(*this)[p[0]]);
    }



  public: // ---- Deprecated ---------------------------------------------------------------------------------



  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      int k=size();
      oss<<indent<<"(";
      for(int i=0; i<k; i++){
	oss<<(*this)[i];
	if(i<k-1) oss<<",";
      }
      oss<<")";//["<<offset<<"]";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GstridesB& x){
      stream<<x.str(); return stream;
    }

  };

}


#endif 
    /*
    GstridesB(const int i0, const int i1): BASE(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }

    GstridesB(const int i0, const int i1, const int i2): BASE(3){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
    }

    GstridesB(const int i0, const int i1, const int i2, const int i3): BASE(4){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
    }

    GstridesB(const int i0, const int i1, const int i2, const int i3, const int i4): BASE(5){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
      (*this)[4]=i4;
    }
    */
    /*
    GstridesB insert(const int d, const int x) const{
      GstridesB r(size()+1,fill_raw());
      for(int i=0; i<d; i++) r[i]=(*this)[i];
      r[d]=x;
      for(int i=d; i<size(); i++) r[i+1]=(*this)[i];
      return r;
    }
    */
   /*
    GstridesB append(const int s) const{
      GstridesB R(*this);
      R.push_back(s);
      return R;//.set_offset(offset);
    }

    GstridesB cat(const GstridesB& y) const{
      GstridesB R(size()+y.size(),fill_raw());
      for(int i=0; i<size(); i++) R[i]=(*this)[i];
      for(int i=0; i<y.size(); i++) R[size()+i]=y[i];
      return R;
    }

    GstridesB prepend(const int i) const{
      if(i<0) return *this;
      GstridesB R;
      R.push_back(i);
      for(auto p:*this) R.push_back(p);
      return R;
    }
    */
    /*
    GstridesB chunk(int beg, int n=-1) const{
      if(beg<0) beg=size()+beg;
      if(n==-1) n=size()-beg;
      GstridesB R(n,fill_raw());
      for(int i=0; i<n; i++)
	R[i]=(*this)[beg+i];
      return R;//.set_offset(offset);
    }
    */
   /*
    GstridesB permute(const vector<int>& p) const{
      CNINE_ASSRT(p.size()<=size());
      GstridesB R;
      R.resize(size());
      for(int i=0; i<p.size(); i++)
	R[i]=(*this)[p[i]];
      for(int i=p.size(); i<size(); i++)
	R[i]=(*this)[i];
      return R;
    }
    */


    //GstridesB(const int k, const fill_zero& dummy): 
    //BASE(k,0){}

    //GstridesB(const initializer_list<size_t>& lst):
    //BASE(lst){}

    //GstridesB(const int i0): 
    //BASE(1){
    //(*this)[0]=i0;
    //}

    //size_t total() const{
    //size_t t=1; 
    //for(int i=0; i<size(); i++) t*=(*this)[i];
    //return t;
    //}

    //bool operator==(const GstridesB& x) const{
    //if(size()!=x.size()) return false;
    //for(int i=0; i<size(); i++)
    //if((*this)[i]!=x[i]) return false;
    //return true;
    //}
    //GstridesB remove(const vector<int>& v) const{
    //return cnine::except(*this,v);
    //}
