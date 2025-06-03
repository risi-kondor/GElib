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

#ifndef _FixedkGatherMap
#define _FixedkGatherMap

#include "Cnine_base.hpp"
#include "Ltensor.hpp"
#include "hlists.hpp"
#include "RemoteCopy.hpp"

namespace cnine{



  class FixedkGatherMap: public Ltensor<int>{
  public:

    typedef Ltensor<int> BASE;

    using BASE::BASE;
    //using TensorView<int>::slice0;

    int in_columns=1;
    int out_columns=1;

    cnine::RemoteCopy<int,BASE> on_device=cnine::RemoteCopy<int,BASE>([this](const int& _dev){
	return to_share(new BASE(*this,_dev));});

  public:

    ~FixedkGatherMap(){
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    FixedkGatherMap(const int _n, const int _K):
      BASE(Gdims({_n,_K+1})){}


  public: // ---- Conversions --------------------------------------------------------------------------------


  public: // ---- Named constructors -------------------------------------------------------------------------


    static FixedkGatherMap random(const int _n, const int _k, const float p){
      FixedkGatherMap r(Gdims({_n,_k+1}));
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<_n; i++){
	r.set(i,0,i);
	for(int j=0; j<_k; j++)
	  if(distr(rndGen)<p)
	    r.set(i,j+1,1);
      }
      return r;
    }


  public: // ---- Transport ----------------------------------------------------------------------------------

    /*
    int* get_arrg(const int _dev=1){
      if(!arrg) make_arrg();
      return arrg;
    }

    void make_arrg(){
      int memsize=arr.memsize+arr.dir.memsize;
      CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
      CUDA_SAFE(cudaMemcpy(arrg, arr.dir.arr, arr.dir.memsize*sizeof(int),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(arrg+arr.dir.memsize, arr.arr, arr.memsize*sizeof(TYPE),cudaMemcpyHostToDevice));  
    }
    */
    
  public: // ---- Access -------------------------------------------------------------------------------------


    int getn() const{
      return dim(0);
    }

    int getk() const{
      return dim(1)-1;
    }

    int size() const{
      return dim(0);
    }

    int target(const int i) const{
      return BASE::operator()(i,0);
    }

    void set_target(const int i, const int x){
      BASE::set(i,0,x);
    }

    int operator()(const int i, const int j) const{
      return BASE::operator()(i,j+1);
    }

    void set(const int i, const int j, const int x){
      BASE::set(i,j+1,x);
    }


    void for_each(std::function<void(const int i, const int j)> lambda) const{
      int N=getn();
      int K=getk();
      for(int i=0; i<N; i++){
	int targt=target(i);
	for(int j=0; j<K; j++)
	  lambda(targt,(*this)(i,j));
      }
    }


  public: // ---- Operations ---------------------------------------------------------------------------------



  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "FixedkGatherMap";
    }

    string repr() const{
      return "FixedkGatherMap";
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<BASE::str(indent);
      return oss.str();
    }


	friend ostream& operator<<(ostream& stream, const FixedkGatherMap& v){
      stream<<v.str(); return stream;}

  };



}

#endif 
