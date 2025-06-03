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

#ifndef _GatherMap
#define _GatherMap

#include "Ptens_base.hpp"
#include "array_pool.hpp"

namespace cnine{


  class GatherList{
  public:

    int* arr=nullptr;
    int n=0;
    int _target=0;
    bool is_view=false;

    ~GatherList(){
      if(is_view) return;
      delete[] arr;
    }

  public: // ---- Constructors -------------------------------------------------------------------------------


    GatherList(int* _arr, const int _n, const int target):
      arr(_arr), n(_n), _target(target), is_view(true){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    GatherList(const GatherList& x){
      n=x.n;
      _target=x._target;
      arr=new int[2*n];
      std::copy(x.arr,x.arr+2*n,arr);
    }

    GatherList(GatherList&& x){
      n=x.n; x.n=0;
      _target=x._target;
      arr=x.arr; x.arr=nullptr;
      is_view=x.is_view;
    }

    GatherList& operator=(const GatherList& x)=delete;
    

  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return n;
    }

    pair<int,float> operator()(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In GatherList::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return pair<int,float>(arr[2*i],*reinterpret_cast<float*>(arr+2*i+1));
    }

    int target() const{
      return _target;
    }

    int src(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In GatherList::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return arr[2*i];
    }

    int weight(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In GatherList::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return *reinterpret_cast<float*>(arr+2*i+1);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<_target<<"<-(";
      for(int i=0; i<n; i++){
	oss<<"("<<src(i)<<","<<weight(i)<<")";
	if(i<n-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const GatherList& v){
      stream<<v.str(); return stream;}

  };



  class GatherMap{
  public:

    int n=0;
    int* arr=nullptr;
    int* arrg=nullptr;
    int dev=0;
    int memsize=0;
    bool is_view=false;

    ~GatherMap(){
      if(is_view) return;
      if(arr) delete[] arr; 
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    GatherMap(const int _n, const int edges, const int _dev=0):
      n(_n), dev(_dev){
      memsize=3*n+2*edges;
      if(dev==0){
	arr=new int[std::max(memsize,1)];
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, (std::max(memsize,1))*sizeof(int)));
      }
    }

    GatherMap(int n, int nedges, const fill_raw& dummy, const int dev=0):
      GatherMap(n,nedges,dev){}
      


  public: // ---- Copying ------------------------------------------------------------------------------------


    GatherMap(const GatherMap& x):
      GatherMap(x.n,(x.memsize-3*x.n)/2,x.dev){
      CNINE_COPY_WARNING();
      if(dev==0){
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
    }

    GatherMap(GatherMap&& x){
      CNINE_MOVE_WARNING();
      n=x.n;
      dev=x.dev;
      memsize=x.memsize;
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
    }

    GatherMap& operator=(const GatherMap& x){
      CNINE_ASSIGN_WARNING();
      n=x.n;
      dev=x.dev;
      memsize=x.memsize;
      if(!is_view){
	if(arr) delete[] arr; 
	if(arrg) {CUDA_SAFE(cudaFree(arrg));}
      }
      if(dev==0){
	arr=new int[std::max(memsize,1)];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, (std::max(memsize,1))*sizeof(int)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
      return *this;
    }

    GatherMap& operator=(GatherMap&& x){
      CNINE_MOVEASSIGN_WARNING();
      if(!is_view){
	if(arr) delete[] arr; 
	if(arrg) {CUDA_SAFE(cudaFree(arrg));}
      }
      n=x.n;
      dev=x.dev;
      memsize=x.memsize;
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      return *this;
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    GatherMap(const GatherMap& x, const int _dev):
      GatherMap(x){
      to_device(_dev);
    }

    GatherMap& to_device(const int _dev){
      if(dev==_dev) return *this;

      if(_dev==0){
	if(dev==1){
	  //cout<<"Moving GatherMap to host "<<endl;
	  delete[] arr;
	  arr=new int[std::max(memsize,1)];
	  CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(int),cudaMemcpyDeviceToHost));  
	  CUDA_SAFE(cudaFree(arrg));
	  arrg=nullptr;
	  dev=0;
	}
      }

      if(_dev>0){
	if(dev==0){
	  //cout<<"Moving GatherMap to device "<<memsize<<endl;
	  if(arrg) CUDA_SAFE(cudaFree(arrg));
	  CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
	  CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(int),cudaMemcpyHostToDevice));  
	  delete[] arr;
	  arr=nullptr;
	  dev=_dev;
	}
      }
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return n;
    }
    
    int target(const int i) const{
      CNINE_CPUONLY();
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In GatherMap::target(const int): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return arr[3*i+2];
    }

    const GatherList operator()(const int i) const{
      CNINE_CPUONLY();
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In GatherMap::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return GatherList(arr+arr[3*i],arr[3*i+1],arr[3*i+2]);
    }

    void for_each(std::function<void(const int, const GatherList)> lambda) const{
      for(int i=0; i<n; i++)
	lambda(target(i),(*this)(i));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GatherMap";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for_each([&](const int i, const GatherList lst){oss<<indent<<i<<": "<<lst<<endl;});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GatherMap& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
    /*
    void push_back(const vector<int>& v){
      int len=2*v.size();
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      for(int i=0; i<len/2; i++){
	arr[tail+2*i]=v[i];
	arr[tail+2*i+1]=1.0;
      }
      dir.push_back(tail,len);
      tail+=len;
    }
    */
