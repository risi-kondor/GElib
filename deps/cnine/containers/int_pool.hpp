/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _int_pool
#define _int_pool

#include "Cnine_base.hpp"


namespace cnine{

  class int_pool{
  public:

    int n;
    int last=-1;
    int memsize;
    int* arr;
    int dev=0;

    ~int_pool(){
      delete[] arr;
    }

    int_pool(const int _n, const int _m):
      n(_n){
      memsize=_n+_m+2;
      arr=new int[memsize];
      arr[0]=n;
      arr[1]=n+2; 
    }
    

  public: // ---- Copying --------------------------------------------------


    int_pool(const int_pool& x):
      n(x.n), last(x.last), memsize(x.memsize){
      arr=new int[memsize];
      std::copy(x.arr,x.arr+memsize,arr);
    }

    int_pool(int_pool&& x):
      n(x.n), last(x.last), memsize(x.memsize){
      arr=x.arr;
      x.arr=nullptr;
    }

    int_pool operator=(const int_pool& x)=delete;


  public: // ---- Transport ------------------------------------------------


    void move_to_device(const int _dev){
      if(dev==_dev) return;
      if(dev==0 &&_dev==1){
	int* arrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
	CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(int),cudaMemcpyHostToDevice));
	delete[] arr;
	arr=arrg;
	dev=1;
      }
      if(dev==1 && _dev==0){
	int* narr=new int[memsize];
	CUDA_SAFE(cudaMemcpy(narr,arr,memsize*sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_SAFE(cudaFree(arr));
	arr=narr;
	dev=0;
      }      
    }



  public: // ---- Access ---------------------------------------------------


    int size() const{
      return n;
    }

    int getn() const{
      return n;
    }

    int tail() const{
      return arr[last+2];
    }

    int addr_of(const int i) const{
      CNINE_ASSRT(i<n);
      return arr[i+1];
    }

    int size_of(const int i) const{
      CNINE_ASSRT(i<n);
      return arr[i+2]-arr[i+1];
    }

    int operator()(const int i, const int j) const{
      CNINE_ASSRT(i<n);
      CNINE_ASSRT(j<size_of(i));
      return arr[arr[i+1]+j];
    }

    int& operator()(const int i, const int j){
      CNINE_ASSRT(i<n);
      CNINE_ASSRT(j<size_of(i));
      return arr[arr[i+1]+j];
    }

    void set(const int i, const int j, const int v){
      CNINE_ASSRT(i<n);
      CNINE_ASSRT(j<size_of(i));
      arr[arr[i+1]+j]=v;
    }

    int add_vec(const int m){
      arr[last+2]=arr[last+1]+m;
      last++;
      return arr[last+1];
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "int_pool";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<n; i++){
	oss<<indent<<i<<":(";
	for(int j=0; j<size_of(i); j++)
	  oss<<(*this)(i,j)<<",";
	if(size_of(i)>0) oss<<"\b";
	oss<<")"<<endl;
      //if(is_labeled) oss<<labels.str(indent+"  ")<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const int_pool& x){
      stream<<x.str(); return stream;}

  };

};

#endif 

