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


#ifndef __LdimsList
#define __LdimsList

#include <functional>

#include "Cnine_base.hpp"
#include "pvector.hpp"
#include "GindexSet.hpp"
#include "Ldims.hpp"
#include "Lbatch.hpp"
#include "Lgrid.hpp"
#include "Larray.hpp"
#include "Gdims.hpp"


namespace cnine{


  class LdimsList: public pvector<Ldims>{
  public:

    ~LdimsList(){
    }

    LdimsList(){}

    LdimsList(const initializer_list<reference_wrapper<Ldims> > _ldims){
      for(auto& p:_ldims)
	push_back(p.get().clone());
    }

    // this takes ownership of the Ldims!
    LdimsList(const initializer_list<Ldims*> _ldims){
      for(auto& p:_ldims)
	push_back(p);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------



  public: // ---- Conversions --------------------------------------------------------------------------------


    LdimsList(const pvector<Ldims>& x):
      pvector<Ldims>(x){}

    LdimsList(pvector<Ldims>&& x):
      pvector<Ldims>(std::move(x)){}


    operator Gdims() const{
      Gdims R;
      for(auto p:*this){
	Gdims D(*p);
	for(auto d:D)
	  R.push_back(d);
      }
      return R;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    int total() const{
      int t=1; for(int i=0; i<size(); i++) t*=(*this)[i]->total();
      return t;
    }


  public: // ---- Batches ------------------------------------------------------------------------------------


    bool is_batched() const{
      if(size()==0) return false;
      if(!dynamic_cast<Lbatch*>((*this)[0])) return false;
      return true;
    }

    int nbatch() const{
      if(!is_batched()) return 1;
      return (*(*this)[0])[0];
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<(*this)[i]->str();
	if(i<size()-1) oss<<",";
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LdimsList& x){
      stream<<x.str(); return stream;}

  };


}

#endif 
    /*
    LdimsList(const vector<Ldims*>& _ldims){
      for(auto& p:_ldims)
	push_back(p->clone());
    }
    */

