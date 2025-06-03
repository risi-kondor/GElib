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


#ifndef _CellMask1r
#define _CellMask1r

#include <map>

#include "Gdims.hpp"


namespace cnine{


  //class CellTlist2: public vector<pair<int,int> >{
  //public:
    //vector<pair<int,int> > lst;
  //};


  class CellMask1r{
  public:

    map<int,vector<int>> lists;
    vector<int> rstrides;
    vector<int> xstrides;

    mutable int n=0;
    mutable int* arrg=nullptr;
    mutable bool current=false;


    ~CellMask1r(){
      for(auto p:lists) delete p.second;
      if(arrg) CUDA_SAFE(cudaFree(arrg));
    }


  public:

    CellMask1r(){}
    
    CellMask1r(const Gdims& rdims, const Gdims& xdims):
      rstrides(rdims.strides()), 
      xstrides(xdims.strides()){
    }


  public:

    CellMask1r(const CellMask1r& x){
      lists=x.lists;
      rstrides=x.rstrides;
      xstrides=x.xstrides;
    }


  public:

    static CellMask1r list(const Rtensor& M, const int nr, const int nx){
      CellMask1r R(Gdims(nr),Gdims(nx)); 
      assert(M.get_k()==2);
      assert(M.dims[1]==2);
      int N=M.dims[0];
      for(int i=0; i<N; i++)
	R.push(Gindex(M(i,0),M(i,1)));
      return R;
    }

    static CellMask1r matrix(const Rtensor& A){
      assert(M.get_k()==2);
      int N=M.dims[0];
      assert(M.dims[1]=N);
      CellMask1r R(Gdims(N),Gdims(N)); 
      for(int i=0; i<N; i++)
	for(int j=0; j<N; j++)
	  if(A(i,j)>0) push(Gindex(i),Gindex(j));
      return R;
    }


  public:
 
    void push(const Gindex& rix, const Gindex& xix){
      current=false;
      int r=rix(rstrides);
      CellTlist2* lst;
      auto it=lists.find(r);
      if(it!=lists.end()) lst=it->second;
      else{
	lst=new CellTlist2();
	lists[r]=lst;
      }
      lst->push_back(xix(xstrides));
    }

    void prepare(const int dev) const{
      if(current) return;
      n=lists.size();
#ifdef _WITH_CUDA
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
#endif
    }

    string str(const string indent=""){
      ostringstream oss;
      for(auto it: lists){
	oss<<indent<<Gindex(it.first,rstrides)<<"<-(";
	//for(auto p:it.second->lst)
	const vector<int>& lst=*it.second;
	for(int i=0; i<lst.size(); i++){
	  oss<<"("<<Gindex(lst[i],xstrides)<<")";
	  if(i<lst.size()-1) oss<<",";
	}
	oss<<")"<<endl;
      }
      return oss.str();
    }

  };


}

#endif 
