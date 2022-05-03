
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3_CGbank
#define _SO3_CGbank

#include <mutex>
#include "SO3_CGcoeffs.hpp"

#define CG_CMEM_DATA_OFFS 4096

#ifdef _WITH_CUDA
#ifdef _DEF_CGCMEM
//__device__ __constant__ unsigned char cg_cmem[CNINE_CONST_MEM_SIZE];
#endif
#ifndef _SO3CG_CUDA_CONCAT
extern __device__ __constant__ unsigned char cg_cmem[]; 
#endif
#endif

namespace GElib{

  class SO3_CGbank{
  private:

    unordered_map<CGindex,SO3_CGcoeffs<float>*> cgcoeffsf;
    unordered_map<CGindex,SO3_CGcoeffs<double>*> cgcoeffsd;
    unordered_map<CGindex,SO3_CGcoeffs<float>*> cgcoeffsfG;
    unordered_map<CGindex,SO3_CGcoeffs<double>*> cgcoeffsdG;
    unordered_map<CGindex,int> cgcoeffsfC;
    
    mutex safety_mx;
    mutex safety_mxC;
    int cmem_index_tail=0;
    int cmem_data_tail=CG_CMEM_DATA_OFFS;
  
  public:

    SO3_CGbank(){}
    
    SO3_CGbank(const SO3_CGbank& x)=delete;
    SO3_CGbank& operator=(const SO3_CGbank& x)=delete;
    
    ~SO3_CGbank(){
      for(auto p:cgcoeffsf) delete p.second;
      for(auto p:cgcoeffsd) delete p.second;
      //for(auto p:cgcoeffsfG) delete p.second; // why is this a problem?
      for(auto p:cgcoeffsdG) delete p.second;
    }
    
    const SO3_CGcoeffs<float>& getf(const CGindex& ix, const int dev=0){
      if(dev==0){
	lock_guard<mutex> lock(safety_mx);
	auto it=cgcoeffsf.find(ix);
	if(it!=cgcoeffsf.end()) return *it->second;
	SO3CG_DEBUG("Computing CG coefficients for "<<ix.str()<<"...");
	SO3_CGcoeffs<float>* r=new SO3_CGcoeffs<float>(ix);
	//lock_guard<mutex> lock(safety_mx);
	//it=cgcoeffsf.find(ix);
	//if(it!=cgcoeffsf.end()) return *it->second;
	cgcoeffsf[ix]=r;
	return *r;
      }else{
	{
	  lock_guard<mutex> lock(safety_mx);
	  auto it=cgcoeffsfG.find(ix);
	  if(it!=cgcoeffsfG.end()) return *it->second;
	}
	SO3_CGcoeffs<float>* r=new SO3_CGcoeffs<float>(getf(ix));
	r->to_device(dev);
	{
	  lock_guard<mutex> lock(safety_mx);
	  auto it=cgcoeffsfG.find(ix);
	  if(it!=cgcoeffsfG.end()) {delete r; return *it->second;}
	  cgcoeffsfG[ix]=r;
	  return *r;
	}
      }
    }

    const SO3_CGcoeffs<double>& getd(const CGindex& ix, const cnine::device& dev=0){
      lock_guard<mutex> lock(safety_mx);
      if(dev.id()==0){
	auto it=cgcoeffsd.find(ix);
	if(it!=cgcoeffsd.end()) return *it->second;
	SO3_CGcoeffs<double>* r=new SO3_CGcoeffs<double>(ix);
	lock_guard<mutex> lock(safety_mx);
	it=cgcoeffsd.find(ix);
	if(it!=cgcoeffsd.end()) return *it->second;
	cgcoeffsd[ix]=r;
	return *r;
      }else{
	auto it=cgcoeffsdG.find(ix);
	if(it!=cgcoeffsdG.end()) return *it->second;
	SO3_CGcoeffs<double>* r=new SO3_CGcoeffs<double>(getd(ix));
	r->to_device(dev);
	lock_guard<mutex> lock(safety_mx);
	it=cgcoeffsdG.find(ix);
	if(it!=cgcoeffsdG.end()) return *it->second;
	cgcoeffsdG[ix]=r;
	return *r;
      }
    }


#ifdef _WITH_CUDA
    int getfC(const int l1, const int l2, const int l){
      lock_guard<mutex> lock(safety_mxC);
      CGindex ix(l1,l2,l);
      auto it=cgcoeffsfC.find(ix);
      if(it!=cgcoeffsfC.end()) return it->second;
      const SO3_CGcoeffs<float>& coeffs=getf(ix);
      //cout<<cmem_index_tail<<": "<<l1<<" "<<l2<<" "<<l<<endl;

      if(cmem_index_tail+4*sizeof(int)>CG_CMEM_DATA_OFFS || cmem_data_tail+sizeof(float)*coeffs.asize>CNINE_CONST_MEM_SIZE){
	//SO3CG_DEBUG("GPU constant memory full. Reverting to storing CG coefficients in global memory.");
	return -128;
      }

      if(cmem_index_tail+4*sizeof(int)>CG_CMEM_DATA_OFFS){
	cerr<<"SO3_CGbank: no room to store index entry in constant memory."<<endl; exit(-1);}
      int ix_entry[4];
      ix_entry[0]=l1;
      ix_entry[1]=l2;
      ix_entry[2]=l;
      ix_entry[3]=cmem_data_tail;
      CUDA_SAFE(cudaMemcpyToSymbol(cg_cmem,reinterpret_cast<void*>(ix_entry),
	  4*sizeof(int),cmem_index_tail,cudaMemcpyHostToDevice));
      cmem_index_tail+=4*sizeof(int);
      cgcoeffsfC[ix]=cmem_data_tail; 
      if(cmem_data_tail+sizeof(float)*coeffs.asize>CNINE_CONST_MEM_SIZE){
	cerr<<"SO3_CGbank: no room to store CG matrix in constant memory."<<endl; exit(-1);}
      //cout<<l1<<l2<<l<<coeffs.arr[0]<<endl;
      CUDA_SAFE(cudaMemcpyToSymbol(cg_cmem,reinterpret_cast<void*>(coeffs.arr),
	  coeffs.asize*sizeof(float),cmem_data_tail,cudaMemcpyHostToDevice));
      int r=cmem_data_tail;
      cmem_data_tail+=sizeof(float)*coeffs.asize;
      SO3CG_DEBUG("GPU constant memory tail: "<<cmem_data_tail);
      return r;
    }
#endif 


#ifdef _WITH_CUDA
    /*
    int getfC(const int l1, const int l2, const int l, const cudaStream_t& stream){
      lock_guard<mutex> lock(safety_mxC);
      CGindex ix(l1,l2,l);
      auto it=cgcoeffsfC.find(ix);
      if(it!=cgcoeffsfC.end()) return it->second;
      const SO3_CGcoeffs<float>& coeffs=getf(ix);
      //cout<<cmem_index_tail<<": "<<l1<<" "<<l2<<" "<<l<<endl;
      if(cmem_index_tail+4*sizeof(int)>CG_CMEM_DATA_OFFS){
	cerr<<"SO3_CGbank: no room to store index entry in constant memory."<<endl; exit(-1);}
      int ix_entry[4];
      ix_entry[0]=l1;
      ix_entry[1]=l2;
      ix_entry[2]=l;
      ix_entry[3]=cmem_data_tail;
      CUDA_SAFE(cudaMemcpyToSymbolAsync(cg_cmem,reinterpret_cast<void*>(ix_entry),
	  4*sizeof(int),cmem_index_tail,cudaMemcpyHostToDevice,stream));
      cmem_index_tail+=4*sizeof(int);
      cgcoeffsfC[ix]=cmem_data_tail; 
      if(cmem_data_tail+sizeof(float)*coeffs.asize>CNINE_CONST_MEM_SIZE){
	cerr<<"SO3_CGbank: no room to store CG matrix in constant memory."<<endl; exit(-1);}
      //cout<<l1<<l2<<l<<coeffs.arr[0]<<endl;
      CUDA_SAFE(cudaMemcpyToSymbolAsync(cg_cmem,reinterpret_cast<void*>(coeffs.arr),
	  coeffs.asize*sizeof(float),cmem_data_tail,cudaMemcpyHostToDevice,stream));
      int r=cmem_data_tail;
      cmem_data_tail+=sizeof(float)*coeffs.asize;
      return r;
    }
    */
#endif 

    template<class TYPE>
    const SO3_CGcoeffs<TYPE>& get(const int l1, const int l2, const int l);

    template<class TYPE>
    const SO3_CGcoeffs<TYPE>& getG(const int l1, const int l2, const int l);

  };


  /*
  template<>
  inline const SO3_CGcoeffs<float>& SO3_CGbank::get<float>(const int l1, const int l2, const int l){
    return getf(CGindex(l1,l2,l));}
  
  template<>
  inline const SO3_CGcoeffs<double>& SO3_CGbank::get<double>(const int l1, const int l2, const int l){
    return getd(CGindex(l1,l2,l));}
  
  template<>
  inline const SO3_CGcoeffs<float>& SO3_CGbank::getG<float>(const int l1, const int l2, const int l){
    return getf(CGindex(l1,l2,l),device_id(1));}
  
  template<>
  inline const SO3_CGcoeffs<double>& SO3_CGbank::getG<double>(const int l1, const int l2, const int l){
    return getd(CGindex(l1,l2,l),device_id(1));}
  */
    

} 

#endif
