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


#ifndef _CnineMultiTensorArray
#define _CnineMultiTensorArray

//#include "CtensorB.hpp"

#include <map>

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename INDEX, typename ARRAY>
  class MultiTensorArray{
  public:

    map<INDEX,ARRAY*> parts;
    bool is_view=false;

    ~MultiTensorArray(){
      if(!is_view) 
	for(auto p: parts) delete p.second;  
    }


  public: // ---- Constructors --------------------------------------------------------------------------------------


    MultiTensorArray(){}


  public: // ---- Copying -------------------------------------------------------------------------------------------


    MultiTensorArray(const MultiTensorArray& x){
      CNINE_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=new ARRAY(*p.second);
    }

    MultiTensorArray(MultiTensorArray&& x){
      CNINE_MOVE_WARNING();
      parts=x.parts;
      x.parts.clear();
    }

    MultiTensorArray& operator=(const MultiTensorArray& x){
      CNINE_ASSIGN_WARNING();
      for(auto p: parts) delete p.second;  
      parts.clear();
      for(auto& p:x.parts)
	parts[p.first]=new ARRAY(*p.second);
      return *this;
    }

    MultiTensorArray& operator=(MultiTensorArray&& x){
      CNINE_MOVEASSIGN_WARNING();
      for(auto p: parts) delete p.second;  
      parts=x.parts;
      x.parts.clear();
      return *this;
    }


  public: // ---- Views -----------------------------------------------------------------------------------------


    MultiTensorArray view(){
      MultiTensorArray R;
      for(auto p: parts){
	R.parts[p.first]=new ARRAY(p.second->ARRAY::view());
      }
      return R;
      
    }


  public: // ---- Transport -----------------------------------------------------------------------------------------


    MultiTensorArray& move_to_device(const int _dev){
      for(auto p:parts)
	p.second->move_to_device(_dev);
      return *this;
    }
    
    /*
    MultiTensorArray to_device(const int _dev) const{
      MultiTensorArray<ARRAY> R;
      for(auto p:parts)
	R.parts.push_back(new ARRAY(p->to_device(_dev)));
      return R;
    }
    */

  public: // ---- Access --------------------------------------------------------------------------------------------
  

    Gdims get_adims() const{
      if(parts.size()>0) return parts.begin()->second->get_adims();
      return 0;
    }

    int get_dev() const{
      if(parts.size()>0) return parts.begin()->second->get_dev();
      return 0;
    }

    int get_device() const{
      if(parts.size()>0) return parts.begin()->first->get_dev();
      return 0;
    }

    


  public: // ---- Operations ---------------------------------------------------------------------------------------


    /*
    MultiTensorArray operator-(const MultiTensorArray& y) const{
      MultiTensorArray R;
      for(int l=0; l<parts.size(); l++){
	R.parts.push_back(new ARRAY((*parts[l])-(*y.parts[l])));
      }
      return R;
    }
    */

  public: // ---- Cumulative Operations -----------------------------------------------------------------------------

    /*
    void operator+=(const MultiTensorArray& x){
      add(x);
    }
    
    void add(const MultiTensorArray& x){
      assert(parts.size()==x.parts.size());
      for(int l=0; l<parts.size(); l++)
	parts[l]->add(*x.parts[l]);
    }


    void add_gather(const MultiTensorArray& x, const cnine::Rmask1& mask){
      assert(parts.size()==x.parts.size());
      for(int l=0; l<parts.size(); l++){
	parts[l]->add_gather(*x.parts[l],mask);
      }
    }
    */
    
    
  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "cnine::MultiTensorArray<"+ARRAY().classname()+">";
    }

    /*
    string str(const string indent="") const{
      ostringstream oss;
	for(int l=0; l<parts.size(); l++){
	  if(!parts[l]) continue;
	  oss<<indent<<"Part l="<<l<<":\n";
	  oss<<parts[l]->str(indent+"  ");
	  oss<<endl;
	}
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const MultiTensorArray& x){
      stream<<x.str(); return stream;
    }
    */


  };

}


#endif 
