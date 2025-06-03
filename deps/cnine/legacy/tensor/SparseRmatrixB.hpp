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


#ifndef _SparseRmatrixB
#define _SparseRmatrixB

#include "Cnine_base.hpp"
#include <map>

#include "Gdims.hpp"
#include "IntTensor.hpp"
#include "RtensorA.hpp"
#include "CSRmatrix.hpp"

// This is a temporary hack for ptens!!!

namespace cnine{


  class SparseVecB: public map<int, float>{
  public:
    int n=0;

    ~SparseVecB(){}


  public:

    SparseVecB(const int _n): n(_n){}

    
  public: // ---- Boolean ----------------------------------------------------------------------------------


    bool operator==(const SparseVecB& x){
      for(auto p: *this){
	auto it=x.find(p.first);
	if(it==x.end()) return false;
	if(p.second!=it->second) return false;
      }
      return true;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    void set(const int i, const float v){
      (*this)[i]=v;
    }

    void forall_nonzero(const std::function<void(const int, const float)>& lambda){
      for(auto p:*this)
	lambda(p.first,p.second);
    }


  public:

    string str(const string indent="") const{
      ostringstream oss;
      for(auto p:*this){
	oss<<"("<<p.first<<","<<p.second<<")";
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SparseVecB& x){
      stream<<x.str(); return stream;}

  };



  class SparseRmatrixB{
  public:

    typedef RtensorA rtensor;

    int n=0;
    int m=0;
    map<int,SparseVecB*> lists;

    //mutable int n=0;
    //mutable int* arrg=nullptr;
    //mutable array_pool<float> pack;
    //mutable bool pack_current=false;


    ~SparseRmatrixB(){
      //for(auto p:lists) delete p.second;
      //if(arrg) CUDA_SAFE(cudaFree(arrg));
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    SparseRmatrixB(){} 

    SparseRmatrixB(const int _n, const int _m): 
      n(_n), m(_m){} 
    

  public: // ---- Copying ------------------------------------------------------------------------------------


    SparseRmatrixB(const SparseRmatrixB& x){
      CNINE_COPY_WARNING();
      n=x.n; 
      m=x.m;
      for(auto p:x.lists) 
	lists[p.first]=new SparseVecB(*p.second);
    }

    SparseRmatrixB(SparseRmatrixB&& x){
      CNINE_MOVE_WARNING();
      n=x.n; 
      m=x.m;
      lists=std::move(x.lists);
      x.lists.clear();
    }

    SparseRmatrixB& operator=(const SparseRmatrixB& x)=delete;


  public: // ---- Named constructors -------------------------------------------------------------------------


    static SparseRmatrixB random_symmetric(const int _n, const float p=0.5){
      SparseRmatrixB G(_n,_n);
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<_n; i++) 
	for(int j=0; j<i; j++)
	  if(distr(rndGen)<p){
	    G.set(i,j,1.0);
	    G.set(j,i,1.0);
	  }
      return G;
    }

    static SparseRmatrixB from_list(const IntTensor& M){
      CNINE_ASSRT(M.ndims()==2);
      CNINE_ASSRT(M.dim(1)==2);
      int n=0; int m=0;
      for(int i=0; i<M.dim(0); i++){
	n=std::max(M(i,0),n);
	m=std::max(M(i,1),m);
      }
      SparseRmatrixB R(n,m); 
      for(int i=0; i<M.dim(0); i++)
	R.set(M(i,0),M(i,1),1);
      return R;
    }

    static SparseRmatrixB from_matrix(const IntTensor& A){
      CNINE_ASSRT(A.ndims()==2);
      int n=A.dim(0);
      int m=A.dim(1);
      SparseRmatrixB R(n,m); 
      for(int i=0; i<n; i++)
	for(int j=0; j<m; j++)
	  if(A(i,j)>0) R.set(i,j,A(i,j));
      return R;
    }


  public: // ---- Conversions ------------------------------------------------------------------------------


    SparseRmatrixB(const rtensor& x){
      CNINE_ASSRT(x.ndims()==2);
      n=x.dim(0);
      m=x.dim(1);
      for(int i=0; i<n; i++)
	for(int j=0; j<m; j++)
	  if(x(i,j)!=0) set(i,j,x(i,j));
    }

    rtensor dense() const{
      auto R=rtensor::zero({n,m});
      forall_nonzero([&](const int i, const int j, const float v){
	  R.set(i,j,v);});
      return R;
    }


  public: // ---- Boolean ----------------------------------------------------------------------------------


    bool operator==(const SparseRmatrixB& x) const{
      for(auto p: lists){
	auto it=x.lists.find(p.first);
	if(it==x.lists.end()) return false;
	if(*p.second!=*it->second) return false;
      }
      return true;
    }


  public: // ---- Access -----------------------------------------------------------------------------------


    int getn() const{
      return n;
    }

    int getm() const{
      return m;
    }

    float operator()(const int i, const int j) const{
      auto r=conditional_rowp(i);
      if(r==nullptr) return 0;
      auto it=r->find(j);
      if(it==r->end()) return 0;
      return it->second;
    }
 
    void set(const int i, const int j, const float v){
      //pack_current=false;
      CNINE_ASSRT(i<n);
      CNINE_ASSRT(j<m);
      auto it=lists.find(i);
      if(it==lists.end()){
	SparseVecB* lst=new SparseVecB(m);
	lists[i]=lst;
	it=lists.find(i);
      }
      it->second->set(j,v);
    }

    SparseVecB& row(const int i){
      CNINE_ASSRT(i<n);
      if(lists.find(i)==lists.end())
	lists[i]=new SparseVecB(m);
      return *lists[i];
    }

    const SparseVecB& row(const int i) const{
      CNINE_ASSRT(i<n);
      if(lists.find(i)==lists.end())
	const_cast<SparseRmatrixB*>(this)->lists[i]=new SparseVecB(m);
      return *(const_cast<SparseRmatrixB*>(this)->lists[i]);
    }

    SparseVecB* conditional_rowp(const int i) const{
      CNINE_ASSRT(i<n);
      auto it=lists.find(i);
      if(lists.find(i)==lists.end()) return nullptr;
      return it->second;
    }

    void forall_nonzero(std::function<void(const int, const int, const float)> lambda) const{
      for(auto& p: lists){
	int i=p.first;
	p.second->forall_nonzero([&](const int j, const float v){
	    lambda(i,j,v);});
      }
    }


  public: // ---- Pack -------------------------------------------------------------------------------------

    /*
    array_pool<float>& get_pack() const{
      if(!pack_current) refresh_pack();
    }

    void refresh_pack() const{
      int t=0; 
      for(auto p:lists) t+=p.second->size();
      pack.reserve(2*t);
      pack.dir.resize
    }
    */

  public: // ---- Operations -------------------------------------------------------------------------------


    SparseRmatrixB transp() const{
      SparseRmatrixB R(m,n);
      forall_nonzero([&](const int i, const int j, const float v){R.set(j,i,v);});
      return R;
    }


    CSRmatrix<float> csrmatrix() const{
      cnine::CSRmatrix<float> R;
      R.dir.resize0(n);
      //cout<<R.dir<<endl;
      int t=0;
      for(auto q:lists) t+=2*(q.second->size());
      R.reserve(t);
      
      t=0;
      for(int i=0; i<n; i++){
	R.dir.set(i,0,t);
	auto it=lists.find(i);
	if(it==lists.end()){
	  R.dir.set(i,1,0);
	  continue;
	}
	const SparseVecB& v=*it->second;
	R.dir.set(i,1,2*v.size());
	for(auto p:v){
	  *reinterpret_cast<int*>(R.arr+t)=p.first;
	  R.arr[t+1]=p.second;
	  t+=2;
	}
      }
      R.tail=t;
      return R;
    }


  public: // ---- GPU side ---------------------------------------------------------------------------------


    /*
    void prepare(const int dev) const{
      //#ifdef _WITH_CUDA
      if(current) return;
      int n=lists.size();
      if(arrg) CUDA_SAFE(cudaFree(arrg));
      int N=n;
      for(auto p:lists)
	N+=2+2*p.second->size();
      int* arr=new int[N];

      int i=0;
      int lp=n;
      for(auto p:lists){
	arr[i]=lp;
	auto lst=*p.second;
	arr[lp]=p.first;
	arr[lp+1]=lst.size();
	for(int j=0; j<lst.size(); j++){
	  arr[lp+2+j]=lst[j];
	}
	lp+=2+lst.size();
	i++;
      }

      CUDA_SAFE(cudaMalloc((void **)&arrg, N*sizeof(int)));
      CUDA_SAFE(cudaMemcpy(arrg,arr,N*sizeof(int),cudaMemcpyHostToDevice));
      delete[] arr;
      //#endif
    }
    */


    public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "SparseRmatrixB";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(auto it: lists){
	oss<<indent<<it.first<<"<-(";
	oss<<*it.second;
	oss<<")"<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SparseRmatrixB& x){
      stream<<x.str(); return stream;}
    
  };

}


namespace std{
  template<>
  struct hash<cnine::SparseVecB>{
  public:
    size_t operator()(const cnine::SparseVecB& x) const{
      size_t t=1;
      for(auto p: x){
	t=(t<<1)^hash<int>()(p.first);
	t=(t<<1)^hash<int>()(p.second);
      }
      return t;
    }
  };
}

namespace std{
  template<>
  struct hash<cnine::SparseRmatrixB>{
  public:
    size_t operator()(const cnine::SparseRmatrixB& x) const{
      size_t t=1;
      for(auto p: x.lists){
	t=(t<<1)^hash<int>()(p.first);
	t=(t<<1)^hash<cnine::SparseVecB>()(*p.second);
      }
      return t;
    }
  };
}


#endif 
