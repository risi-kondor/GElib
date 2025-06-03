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


#ifndef _CnineCtensorB_multiArray
#define _CnineCtensorB_multiArray

#include "CtensorB.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif 

#ifdef _WITH_CUBLAS
#include <cublas_v2.h>
extern cublasHandle_t cnine_cublas;
#endif 


namespace cnine{

  template<typename ARRAY>
  class CtensorB_multiArray{
  public:

    vector<ARRAY*> parts;
    bool is_view=false;

    ~CtensorB_multiArray(){
      if(!is_view) 
	for(auto p: parts) delete p;  
    }


  public: // ---- Constructors --------------------------------------------------------------------------------------


    CtensorB_multiArray(){}


  public: // ---- Copying -------------------------------------------------------------------------------------------


    CtensorB_multiArray(const CtensorB_multiArray& x){
      CNINE_COPY_WARNING();
      for(auto& p:x.parts)
	parts.push_back(new ARRAY(*p));
    }

    CtensorB_multiArray(CtensorB_multiArray&& x){
      CNINE_MOVE_WARNING();
      parts=x.parts;
      x.parts.clear();
    }

    CtensorB_multiArray& operator=(const CtensorB_multiArray& x){
      CNINE_ASSIGN_WARNING();
      for(auto p: parts) delete p;  
      parts.clear();
      for(auto& p:x.parts)
	parts.push_back(new ARRAY(*p));
      return *this;
    }

    CtensorB_multiArray& operator=(CtensorB_multiArray&& x){
      CNINE_MOVEASSIGN_WARNING();
      for(auto p: parts) delete p;  
      parts=x.parts;
      x.parts.clear();
      return *this;
    }


  public: // ---- Views -----------------------------------------------------------------------------------------


    CtensorB_multiArray view(){
      CtensorB_multiArray R;
      for(auto p: parts){
	R.parts.push_back(new ARRAY(p->ARRAY::view()));
      }
      // ifdef WITH_FAKE_GRAD
      // if(grad) R.grad=new SO3vecB(grad->view());
      // endif 
      return R;
      
    }


  public: // ---- Transport -----------------------------------------------------------------------------------------


    CtensorB_multiArray& move_to_device(const int _dev){
      for(auto p:parts)
	p->move_to_device(_dev);
      return *this;
    }
    
    CtensorB_multiArray to_device(const int _dev) const{
      CtensorB_multiArray<ARRAY> R;
      for(auto p:parts)
	R.parts.push_back(new ARRAY(p->to_device(_dev)));
      return R;
    }


  public: // ---- Access --------------------------------------------------------------------------------------------
  

    Gdims get_adims() const{
      if(parts.size()>0) return parts[0]->get_adims();
      return Gdims({0});
    }

    int get_dev() const{
      if(parts.size()>0) return parts[0]->get_dev();
      return 0;
    }

    int get_device() const{
      if(parts.size()>0) return parts[0]->get_dev();
      return 0;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------------


    CtensorB_multiArray operator-(const CtensorB_multiArray& y) const{
      CtensorB_multiArray R;
      for(int l=0; l<parts.size(); l++){
	R.parts.push_back(new ARRAY((*parts[l])-(*y.parts[l])));
      }
      return R;
    }


  public: // ---- Cumulative Operations -----------------------------------------------------------------------------


    void operator+=(const CtensorB_multiArray& x){
      add(x);
    }
    
    void add(const CtensorB_multiArray& x){
      assert(parts.size()==x.parts.size());
      for(int l=0; l<parts.size(); l++)
	parts[l]->add(*x.parts[l]);
    }


    void add_gather(const CtensorB_multiArray& x, const cnine::Rmask1& mask){
      assert(parts.size()==x.parts.size());
      for(int l=0; l<parts.size(); l++){
	parts[l]->add_gather(*x.parts[l],mask);
      }
    }
    
    
  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "cnine::CtensorB_multiArray<"+ARRAY().classname()+">";
    }

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

    friend ostream& operator<<(ostream& stream, const CtensorB_multiArray& x){
      stream<<x.str(); return stream;
    }


  };

}


#endif 
