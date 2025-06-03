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

/* ABANDONED */

#ifndef _CSRmatrixB
#define _CSRmatrixB

#include "Cnine_base.hpp"
#include "array_pool.hpp"

namespace cnine{

  template<class TYPE>
  class CSRmatrixB{
  public:

    TYPE* arr=nullptr;
    TYPE* arrg=nullptr;
    int memsize=0;
    int tail=0;
    int dev=0;
    bool is_view=false;

    IntTensor dir;
    int n0=0;
    int n1=0;

    CSRmatrixB(){
     if(is_view) return;
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Memory management --------------------------------------------------------------------------


    void reserve(const int n1){
      if(n<=memsize) return;
      int newsize=n;
      if(dev==0){
	TYPE* newarr=new TYPE[newsize];
	if(arr) {std::copy(arr,arr+memsize,newarr); delete[] arr;}
	arr=newarr;
	memsize=newsize;
      }
      if(dev==1){
	TYPE* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, newsize*sizeof(TYPE)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	memsize=newsize;
      }
    }


  public: // ---- Copying ------------------------------------------------------------------------------------

    void init_from(const CSRmatriB& x){
      dev=x.dev;
      tail=x.tail;
      memsize=tail;
      n0=x.n0;
      n1=x.n1;
    }
      

    CSRmatrixB(const CSRmatrixB& x): dir(x.dir){
      CNINE_COPY_WARNING();
      init_from(x);
      if(dev==0){
	arr=new TYPE[memsize];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
    }

    CSRmatrixB(CSRmatrixB&& x): dir(std::move(x.dir)){
      CNINE_MOVE_WARNING();
      init_from(x);
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      is_view=x.is_view;
    }

    CSRmatrixB<TYPE>& operator=(const CSRmatrixB<TYPE>& x){
      CNINE_ASSIGN_WARNING();
      init_from(x);

      if(!is_view){
	if(arr) delete arr; 
	if(arrg){CUDA_SAFE(cudaFree(arrg));}
      }
      arr=nullptr;
      arrg=nullptr;
      is_view=false;

      if(dev==0){
	arr=new TYPE[memsize]; 
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, memsize*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
      return *this;
    }


    CSRmatrixB<TYPE>& operator=(CSRmatrixB<TYPE>&& x){
      CNINE_MOVEASSIGN_WARNING();
      init_from(x);
      if(!is_view){
	delete arr; 
	if(arrg){CUDA_SAFE(cudaFree(arrg));}
      }
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      tail=x.tail;
      memsize=x.memsize;
      dir=std::move(x.dir);
      is_view=x.is_view;
      return *this;
    }

  
  public: // ---- Access -------------------------------------------------------------------------------------


    int getn() const{
      return n0;
    }

    int getm() const{
      return n1;
    }

    int row_offs(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::CSRmatrixB: row index "+to_string(i)+" out of range (0,"+to_string(n0-1)+")."));
      return dir(i);
    }

    int row_size(const int i) const{
      CNINE_CHECK_RANGE(if(i<0 || i>=n0) 
	  throw std::out_of_range("cnine::CSRmatrixB: row index "+to_string(i)+" out of range (0,"+to_string(n0-1)+")."));
      return dir(2*i+1);
    }

    int ix(const int i, const int j) const{
      int offs=row_size(i);
      int n=row_size(i);
      for(int a=0; a<n; a++)
	if(dir(offs+a)==j) return 

	   TYPE operator()(const int i0, const int i1){
      CNINE_CHECK_RANGE(if(i0<0 || i1<0 || i0>=n0 || i1>=n1) 
	  throw std::out_of_range("cnine::CSRmatrixB: index "+Gindex({i0,i1}).str()+" out of range of size "+Gdims({n0,n1}).str()));
      
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for_each([&](const int i, const svec<TYPE> lst){oss<<indent<<i<<": "<<lst<<endl;});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CSRmatrixB& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
 

  /*
    const svec<TYPE> operator()(const int i) const{
      CNINE_CPUONLY();
      CNINE_CHECK_RANGE(if(i>=size()) throw std::out_of_range("In CSRmatrixB::operator(): index "+to_string(i)+" out of range (0,"+to_string(size()-1)+")."));
      return svec<TYPE>(arr+dir(i,0),dir(i,1)/2);
    }

    void for_each(std::function<void(const int, const svec<TYPE>)> lambda) const{
      for(int i=0; i<size(); i++)
	lambda(i,(*this)(i));
    }
    */

    /*
    void push_back(const vector<int>& ix, const vector<TYPE>& v){
      int len=ix.size();
      CNINE_ASSRT(v.size()==len);
      if(tail+2*len>memsize)
	reserve(std::max(2*memsize,tail+2*len));
      for(int i=0; i<len; i++){
	arr[tail+2*i]=ix[i];
	arr[tail+2*i+1]=v[i];
      }
      dir.push_back(tail,2*len);
      tail+=2*len;
    }
    */
