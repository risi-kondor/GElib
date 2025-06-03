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


#ifndef _SparseRmatrix
#define _SparseRmatrix

#include "Cnine_base.hpp"
#include <map>

#include "Gdims.hpp"
#include "IntTensor.hpp"
#include "TensorView.hpp"
#include "CSRmatrix.hpp"
#include "flog.hpp"


namespace cnine{


  class SparseVec: public map<int, float>{
  public:
    int n=0;

    ~SparseVec(){}


  public:

    SparseVec(const int _n): n(_n){}

    
  public: // ---- Boolean ----------------------------------------------------------------------------------


    bool operator==(const SparseVec& x){
      for(auto p: *this){
	auto it=x.find(p.first);
	if(it==x.end()) return false;
	if(p.second!=it->second) return false;
      }
      return true;
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int getn(){
      return n;
    }

    void set(const int i, const float v){
      (*this)[i]=v;
    }

    void forall_nonzero(const std::function<void(const int, const float)>& lambda){
      for(auto p:*this)
	lambda(p.first,p.second);
    }

    size_t rmemsize() const{
      return size()*(sizeof(int)+sizeof(float));
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(auto p:*this){
	oss<<"("<<p.first<<","<<p.second<<")";
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SparseVec& x){
      stream<<x.str(); return stream;}

  };



  class SparseRmatrix{
  public:

    int n=0;
    int m=0;
    map<int,SparseVec*> lists;

    //mutable int n=0;
    //mutable int* arrg=nullptr;
    //mutable array_pool<float> pack;
    //mutable bool pack_current=false;


    ~SparseRmatrix(){
      for(auto p:lists) delete p.second;
      //if(arrg) CUDA_SAFE(cudaFree(arrg));
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    SparseRmatrix(){} 

    SparseRmatrix(const int _n, const int _m): 
      n(_n), m(_m){} 
    

  public: // ---- Copying ------------------------------------------------------------------------------------


    SparseRmatrix(const SparseRmatrix& x){
      //cnine::flog log("SparseRmatrix::SparseRmatrix");
      CNINE_COPY_WARNING();
      n=x.n; 
      m=x.m;
      for(auto p:x.lists) 
	lists[p.first]=new SparseVec(*p.second);
    }

    SparseRmatrix(SparseRmatrix&& x){
      CNINE_MOVE_WARNING();
      n=x.n; 
      m=x.m;
      lists=std::move(x.lists);
      x.lists.clear();
    }

    SparseRmatrix& operator=(const SparseRmatrix& x)=delete;


  public: // ---- Named constructors -------------------------------------------------------------------------


    static SparseRmatrix random_symmetric(const int _n, const float p=0.5){
      SparseRmatrix G(_n,_n);
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<_n; i++) 
	for(int j=0; j<i; j++)
	  if(distr(rndGen)<p){
	    G.set(i,j,1.0);
	    G.set(j,i,1.0);
	  }
      return G;
    }

    static SparseRmatrix from_list(const IntTensor& M){
      CNINE_ASSRT(M.ndims()==2);
      CNINE_ASSRT(M.dim(1)==2);
      int n=0; int m=0;
      for(int i=0; i<M.dim(0); i++){
	n=std::max(M(i,0),n);
	m=std::max(M(i,1),m);
      }
      SparseRmatrix R(n,m); 
      for(int i=0; i<M.dim(0); i++)
	R.set(M(i,0),M(i,1),1);
      return R;
    }

    static SparseRmatrix from_matrix(const IntTensor& A){
      CNINE_ASSRT(A.ndims()==2);
      int n=A.dim(0);
      int m=A.dim(1);
      SparseRmatrix R(n,m); 
      for(int i=0; i<n; i++)
	for(int j=0; j<m; j++)
	  if(A(i,j)>0) R.set(i,j,A(i,j));
      return R;
    }


  public: // ---- Conversions ------------------------------------------------------------------------------


    template<typename TYPE>
    SparseRmatrix(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==2);
      n=x.dim(0);
      m=x.dim(1);
      for(int i=0; i<n; i++)
	for(int j=0; j<m; j++)
	  if(x(i,j)!=0) set(i,j,x(i,j));
    }

//     SparseRmatrix(const rtensor& x){
//       CNINE_ASSRT(x.ndims()==2);
//       n=x.dim(0);
//       m=x.dim(1);
//       for(int i=0; i<n; i++)
// 	for(int j=0; j<m; j++)
// 	  if(x(i,j)!=0) set(i,j,x(i,j));
//     }

    TensorView<float> dense() const{
      Tensor<float> R({n,m},0,0);
      forall_nonzero([&](const int i, const int j, const float v){
	  R.set(i,j,v);});
      return R;
    }

//     rtensor dense() const{
//       auto R=rtensor::zero({n,m});
//       forall_nonzero([&](const int i, const int j, const float v){
// 	  R.set(i,j,v);});
//       return R;
//     }


  public: // ---- Boolean ----------------------------------------------------------------------------------


    bool operator==(const SparseRmatrix& x) const{
      if(n!=x.n || m!=x.m) return false;
      if(lists.size()!=x.lists.size()) return false;
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

    int size() const{
      int t=0;
      for(auto& p: lists)
	t+=p.second->size();
      return t;
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
	SparseVec* lst=new SparseVec(m);
	lists[i]=lst;
	it=lists.find(i);
      }
      it->second->set(j,v);
    }

    SparseVec& row(const int i){
      CNINE_ASSRT(i<n);
      if(lists.find(i)==lists.end())
	lists[i]=new SparseVec(m);
      return *lists[i];
    }

    const SparseVec& row(const int i) const{
      CNINE_ASSRT(i<n);
      if(lists.find(i)==lists.end())
	const_cast<SparseRmatrix*>(this)->lists[i]=new SparseVec(m);
      return *(const_cast<SparseRmatrix*>(this)->lists[i]);
    }

    SparseVec* conditional_rowp(const int i) const{
      CNINE_ASSRT(i<n);
      auto it=lists.find(i);
      if(lists.find(i)==lists.end()) return nullptr;
      return it->second;
    }

    size_t rmemsize() const{
      size_t t=lists.size()*(sizeof(int)+sizeof(SparseVec*));
      for(auto& p: lists)
	t+=p.second->rmemsize();
      return t;
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


    SparseRmatrix transp() const{
      SparseRmatrix R(m,n);
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
	const SparseVec& v=*it->second;
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
      return "SparseRmatrix";
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

    friend ostream& operator<<(ostream& stream, const SparseRmatrix& x){
      stream<<x.str(); return stream;}
    
  };

}


namespace std{
  template<>
  struct hash<cnine::SparseVec>{
  public:
    size_t operator()(const cnine::SparseVec& x) const{
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
  struct hash<cnine::SparseRmatrix>{
  public:
    size_t operator()(const cnine::SparseRmatrix& x) const{
      size_t t=1;
      for(auto p: x.lists){
	t=(t<<1)^hash<int>()(p.first);
	t=(t<<1)^hash<cnine::SparseVec>()(*p.second);
      }
      return t;
    }
  };
}


#endif 
