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


#ifndef _CnineNewTensor
#define _CnineNewTensor

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "DimLabels.hpp"


namespace cnine{

  class NewTensor{
  public:

    Gdims adims;
    Gdims ddims;
    int nbatch=0;
    int _fcode=0;
    int _dev=0;


  public: // ---- Construction ------------------------------------------------------------------------------


    NewTensor& batch(const int b) {nbatch=b; return *this;}
    NewTensor& batches(const int b) {nbatch=b; return *this;}

    NewTensor& blocks(const initializer_list<int>& v) {adims=Gdims(v); return *this;}
    NewTensor& blocks(const vector<int>& v) {adims=Gdims(v); return *this;}
    NewTensor& blocks(const Gdims& v) {adims=v; return *this;}

    NewTensor& array(const initializer_list<int>& v) {adims=Gdims(v); return *this;}
    NewTensor& array(const vector<int>& v) {adims=Gdims(v); return *this;}
    NewTensor& array(const Gdims& v) {adims=v; return *this;}

    NewTensor& dims(const initializer_list<int>& v) {ddims=Gdims(v); return *this;}
    NewTensor& dims(const vector<int>& v) {ddims=Gdims(v); return *this;}
    NewTensor& dims(const Gdims& v) {ddims=v; return *this;}
    
    NewTensor& matrix(const int _n, const int _m) {ddims=Gdims(_n,_m); return *this;}

    NewTensor& zero() {_fcode=0; return *this;}
    NewTensor& raw() {_fcode=1; return *this;}
    NewTensor& ones() {_fcode=2; return *this;}
    NewTensor& sequential() {_fcode=3; return *this;}
    NewTensor& gaussian() {_fcode=4; return *this;}

    NewTensor& dev(const int i) {_dev=i; return *this;}


  public: // ---- Access ------------------------------------------------------------------------------------


    Gdims get_dims() const{
      Gdims r;
      if(nbatch>0) r.extend({nbatch});
      if(adims.size()!=0) r.extend(adims);
      if(ddims.size()!=0) r.extend(ddims);
      return r;
    }

    DimLabels get_labels() const{
      DimLabels r;
      if(nbatch>0) r._batched=true;
      r._narray=adims.size();
      return r;
    }

    int get_fcode() const{
      return _fcode;
    }

    int get_dev() const{
      return _dev;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str() const{
      ostringstream oss;
      oss<<"[ ";
      if(nbatch) oss<<"batch="<<nbatch<<" ";
      if(adims.size()>0) oss<<"array="<<nbatch<<" ";
      if(ddims.size()>0) oss<<"dims="<<nbatch<<" ";
      if(_dev>0) oss<<"dev="<<_dev<<" ";
      oss<<"]";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const NewTensor& x){
      stream<<x.str(); return stream;
    }

  };


  /*
  template<typename CLASS>
  class New: public CLASS::SPEC{
  public:
    New(){}
    operator CLASS(){
      return CLASS(*this);
    }
  };
  */



}

#endif 
