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


#ifndef _Rmask1
#define _Rmask1

#include <map>

#include "Gdims.hpp"
#include "Rtensor2_view.hpp"

namespace cnine{


  //class CellTlist2: public vector<pair<int,int> >{
  //public:
    //vector<pair<int,int> > lst;
  //};


  class Rmask1{
  public:

    int N0=0;
    int M0=0;

    map<int,vector<pair<int,float> > > lists;

    mutable float* arrg=nullptr;
    mutable int* ptrg=nullptr;
    mutable bool current=false;
    mutable bool inv_current=false;
    
    mutable Rmask1* inverse=nullptr;


    ~Rmask1(){
      //for(auto p:lists) delete p.second;
      if(arrg) CUDA_SAFE(cudaFree(arrg));
      if(ptrg) CUDA_SAFE(cudaFree(ptrg));
    }


  public:

    Rmask1(){}
    
    //Rmask1(const Gdims& rdims, const Gdims& xdims):
    //rstrides(rdims.strides()), 
    //xstrides(xdims.strides()){
    //}


  public:


    Rmask1(const Rmask1& x):
      lists(x.lists){
    }

    Rmask1(Rmask1&& x):
      lists(std::move(x.lists)){
      arrg=x.arrg; x.arrg=nullptr;
      ptrg=x.ptrg; x.ptrg=nullptr;
      current=x.current;
      x.current=false;
    }


  public:

    /*
    static Rmask1 list(const Rtensor& M, const int nr, const int nx){
      Rmask1 R(Gdims(nr),Gdims(nx)); 
      assert(M.get_k()==2);
      assert(M.dims[1]=2);
      int N=M.dims[0];
      for(int i=0; i<N; i++)
	R.push(Gindex(M(i,0),M(i,1)));
      return R;
    }
    */

    static Rmask1 matrix(const Rtensor2_view& M){
      int N0=M.n0;
      int M0=M.n1;
      Rmask1 R; 
      for(int i=0; i<N0; i++)
	for(int j=0; j<M0; j++)
	  if(M(i,j)!=0) R.push(i,j,M(i,j));
      return R;
    }


  public:

 
    void push(const int i, const int j, const float v){
      if(i>=N0) N0=i+1;
      if(j>=M0) M0=j+1;
      current=false;
      inv_current=false;
      lists[i].push_back(pair<int,float>(j,v));
    }

    const Rmask1& inv() const{
      if(!inverse || !inv_current) make_inverse();
      return *inverse;
    }


  public:

    void prepare(const int dev=1) const{
      if(current) return;

#ifdef _WITH_CUDA

      int memsize=0;
      for(auto& p:lists)
	memsize+=2+2*p.second.size();
      float* arr=new float[memsize];

      int n=lists.size();
      int* ptr=new int[n];
      //for(int i=0; i<n; i++) ptr[i]=-1;

      int head=0;
      int i=0;
      for(auto& p:lists){
	auto& lst=p.second;
	const int m=lst.size();
	arr[head]=p.first;
	arr[head+1]=m;
	ptr[i++]=head;
	for(int j=0; j<m; j++){
	  arr[head+2+2*j]=lst[j].first;
	  arr[head+2+2*j+1]=lst[j].second;
	}
	head+=2+2*m;
      }

      if(arrg) CUDA_SAFE(cudaFree(arrg));
      CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(float)));
      CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(float),cudaMemcpyHostToDevice));
      delete[] arr;

      if(ptrg) CUDA_SAFE(cudaFree(ptrg));
      CUDA_SAFE(cudaMalloc((void **)&ptrg, n*sizeof(int)));
      CUDA_SAFE(cudaMemcpy(ptrg,ptr,n*sizeof(int),cudaMemcpyHostToDevice));
      delete[] ptr;

#endif
    }


  private:


    void make_inverse() const{
      if(inverse) delete inverse;
      inverse=new Rmask1();
      for(auto& p:lists){
	int i=p.first;
	for(auto q:p.second)
	  inverse->push(q.first,i,q.second);
      }
      inv_current=true;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(auto it: lists){
	oss<<indent<<"X["<<it.first<<"] <- ";
	//for(auto p:it.second->lst)
	auto& lst=it.second;
	for(int i=0; i<lst.size(); i++){
	  oss<<lst[i].second<<"*Y["<<lst[i].first<<"]";
	  if(i<lst.size()-1) oss<<"+";
	}
	oss<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const Rmask1& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif 
