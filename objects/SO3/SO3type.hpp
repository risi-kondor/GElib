
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3type
#define _SO3type

#include "GElib_base.hpp"
#include "GdimsPack.hpp"
#include "CtensorPackObj.hpp"

namespace GElib{

  class SO3type: public vector<int>{
  public:

    SO3type(){};

    //SO3type(const int L): vector<int>(L+1,0){}

    SO3type(const cnine::size_spec& _size): 
      vector<int>(_size.n,0){}
    
    SO3type(const vector<int> list): vector<int>(list){}

    SO3type(const initializer_list<int> list): vector<int>(list){}

    /*
    SO3type static SO3fourier(const int L){
      SO3type tau(L);
      for(int l=0; l<=L; l++)
	tau[l]=2*l+1;
      return tau;
    }
    */


  public:

    int getL() const{
      return this->size()-1;
    }

    int maxl() const{
      return this->size()-1;
    }

    int getM() const{
      return 2*(this->size()-1)+1;
    }

    int operator()(const int l) const{
      return (*this)[l];
    }
    
    void set(const int l, const int m){
      if(l>=size()) resize(l+1);
      (*this)[l]=m;
    }

    void inc(const int l, const int m){
      if(l>=size()) resize(l+1);
      (*this)[l]+=m;
    }

    bool operator<(const SO3type& x){
      if(x.size()>size()) return true;
      if(x.size()<size()) return false;
      for(int i=size()-1; i>=0; i++)
	if((*this)[i]!=x[i]) return (*this)[i]<x[i];
      return false;
    }
    
    int total() const{
      int t=0; for(auto p:*this) t+=p;
      return t;
    }

    vector<int> offs() const{
      vector<int> r;
      int t=0; for(auto p: *this) {r.push_back(t); t+=p;}
      return r;
    }

    void operator+=(const SO3type& x){
      if(x.size()>size()) resize(x.size());
      for(int l=0; l<size() && l<x.size(); l++) (*this)[l]+=x[l];
    }

    
  public:

    static SO3type left(const cnine::GdimsPack& dimsp){
      SO3type R(cnine::size_spec(dimsp.size()));
      for(int i=0; i<dimsp.size(); i++){
	assert(dimsp[i].size()==2);
	R[i]=dimsp[i][0];
      }
      return R;
    }

    static SO3type left(const cnine::CtensorPackObj& x){
      SO3type R(x.tensors.size());
      for(int i=0; i<x.tensors.size(); i++){
	//assert(dimsp[i].size()==2);
	R[i]=x.get_dims(i)[0];
      }
      return R;
    }

    static SO3type TensorProduct(const SO3type& t1, const SO3type& t2){
      SO3type tau(cnine::size_spec(t1.maxl()+t2.maxl()+1));
      for(int l1=0; l1<=t1.maxl(); l1++)
	for(int l2=0; l2<=t2.maxl(); l2++)
	  for(int l=std::abs(l2-l1); l<=l1+l2; l++)
	    tau[l]+=t1(l1)*t2(l2);
      return tau;
    }

    static SO3type TensorProduct(const SO3type& t1, const SO3type& t2, const int maxL){
      SO3type tau(cnine::size_spec(std::min(t1.maxl()+t2.maxl(),maxL)+1));
      for(int l1=0; l1<=t1.maxl(); l1++)
	for(int l2=0; l2<=t2.maxl(); l2++)
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=maxL; l++)
	    tau[l]+=t1(l1)*t2(l2);
      return tau;
    }

    static SO3type TensorSquare(const SO3type& t){
      SO3type tau(cnine::size_spec(2*t.maxl()+1));
      for(int l1=0; l1<=t.maxl(); l1++)
	for(int l2=l1; l2<=t.maxl(); l2++)
	  for(int l=std::abs(l2-l1); l<=l1+l2; l++)
	    if(l1==l2) tau[l]+=t(l1)*(t(l2)+1)/2-t(l1)*((l1+l2-l)%2);
	    else tau[l]+=t(l1)*t(l2);
      return tau;
    }


  public:

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"(";
      if(size()>0) for(int i=0; i<size()-1; i++) oss<<(*this)[i]<<",";
      if(size()>0) oss<<(*this)[size()-1];
      oss<<")";
      return oss.str();
    }

    string repr(const string indent="") const{
      return indent+"<GElib::SO3type"+str()+">";
    }

    friend ostream& operator<<(ostream& stream, const GElib::SO3type& x){
      stream<<x.str(); return stream;}

  };


  // ---- Post-class functions -------------------------------------------------------------------------------


  inline SO3type CGproduct(const SO3type& t1, const SO3type& t2, int _maxl=-1){
    if(_maxl==-1) _maxl=1000;
    SO3type tau(cnine::size_spec(std::min(t1.maxl()+t2.maxl(),_maxl)+1));
    for(int l1=0; l1<=t1.maxl(); l1++)
      for(int l2=0; l2<=t2.maxl(); l2++)
	for(int l=std::abs(l2-l1); l<=l1+l2 && l<=_maxl; l++)
	  tau[l]+=t1(l1)*t2(l2);
    return tau;
  }
  
  inline SO3type CGproduct(const SO3type& t1, const SO3type& t2,  const SO3type& t3, int _maxl=-1){
    if(_maxl==-1) _maxl=1000;
    SO3type a=CGproduct(t1,t2,_maxl+t3.maxl());
    return CGproduct(a,t3,_maxl);

  }

  inline SO3type CGproduct(const SO3type& t1, const SO3type& t2,  const SO3type& t3, const SO3type& t4, int _maxl=-1){
    if(_maxl==-1) _maxl=1000;
    SO3type a=CGproduct(t1,t2,t3,_maxl+t4.maxl());
    return CGproduct(a,t4,_maxl);
  }

  
  inline SO3type CGproduct(const std::vector<SO3type>& v, const int maxl){
    assert(v.size()>1);
    SO3type R=CGproduct(v[0],v[1],maxl);
    for(int i=2; i<v.size(); i++)
      R=CGproduct(R,v[i]);
    return R;
  }

  inline SO3type DiagCGproduct(const SO3type& t1, const SO3type& t2, int _maxl=-1){
    if(_maxl==-1) _maxl=1000;
    SO3type tau(cnine::size_spec(std::min(t1.maxl()+t2.maxl(),_maxl)+1));
    for(int l1=0; l1<=t1.maxl(); l1++)
      for(int l2=0; l2<=t2.maxl(); l2++)
	for(int l=std::abs(l2-l1); l<=l1+l2 && l<=_maxl; l++)
	  tau[l]+=t1(l1);
    return tau;
  }
  
  inline SO3type BlockedCGproduct(const SO3type& t1, const SO3type& t2, const int bsize, int _maxl=-1){
    if(_maxl==-1) _maxl=1000;
    SO3type tau(cnine::size_spec(std::min(t1.maxl()+t2.maxl(),_maxl)+1));
    for(int l1=0; l1<=t1.maxl(); l1++)
      for(int l2=0; l2<=t2.maxl(); l2++)
	for(int l=std::abs(l2-l1); l<=l1+l2 && l<=_maxl; l++)
	  tau[l]+=t1(l1)*bsize;
    return tau;
  }
  
  inline SO3type DDiagCGproduct(const SO3type& t, int _maxl=-1){
    if(_maxl==-1) _maxl=1000;
    SO3type tau(cnine::size_spec(std::min(t.maxl()+t.maxl()%2,_maxl)+1));
    for(int l=0; l+l%2<=tau.maxl(); l++)
      tau[l+l%2]+=t[l];
    return tau;
  }
  
  inline SO3type CGsquare(const SO3type& t, int _maxl=-1){
    if(_maxl==-1) _maxl=1000;
    SO3type tau(cnine::size_spec(std::min(2*t.maxl(),_maxl)+1));
    for(int l1=0; l1<=t.maxl(); l1++){
      for(int l=0; l<=2*l1 && l<=_maxl; l++)
	tau[l]+=(t(l1)*(t(l1)-1))/2+t(l1)*(1-l%2);
      for(int l2=l1+1; l2<=t.maxl(); l2++)
	for(int l=std::abs(l2-l1); l<=l1+l2 && l<=_maxl; l++)
	  tau[l]+=t(l1)*t(l2);
    }
    return tau;
  }
  



  inline cnine::GdimsPack dims(const SO3type& tau1, const SO3type& tau2){
    const int n=std::min(tau1.size(),tau2.size());
    cnine::GdimsPack R;
    for(int i=0; i<n; i++)
      R.push_back(cnine::Gdims(tau1[i],tau2[i]));
    return R;
  }


}


namespace std{
  template<>
  struct hash<GElib::SO3type>{
  public:
    size_t operator()(const GElib::SO3type& tau) const{
      size_t h=0;
      for(auto p:tau)
      h=(h<<1)^hash<int>()(p);
      return h;
    }
  };
}




#endif

