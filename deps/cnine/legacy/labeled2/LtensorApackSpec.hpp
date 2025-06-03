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

#ifndef _CnineLtensorApackSpec
#define _CnineLtensorApackSpec

#include "Cnine_base.hpp"
#include "Gdims.hpp"
#include "DimLabels.hpp"


namespace cnine{

  template<typename KEY, typename SPEC>
  class LtensorApackSpecBase{
  public:

    int nbatch=0;
    Gdims gdims;
    map<KEY,Gdims> ddims;
    int fcode=0;
    int _dev=0;


    LtensorApackSpecBase(){}

    LtensorApackSpecBase(const int _nbatch, const Gdims& _gdims, const map<KEY,Gdims>& _ddims, const int __dev=0):
      nbatch(_nbatch),
      gdims(_gdims),
      ddims(_ddims),
      _dev(__dev){}


  public: // ---- Construction ------------------------------------------------------------------------------


    SPEC batch(const int b) {nbatch=b; return static_cast<SPEC&>(*this);}

    SPEC grid(const initializer_list<int>& v) {gdims=Gdims(v); return static_cast<SPEC&>(*this);}
    SPEC grid(const vector<int>& v) {gdims=Gdims(v); return static_cast<SPEC&>(*this);}
    SPEC grid(const Gdims& v) {gdims=v; return static_cast<SPEC&>(*this);}

    SPEC dims(const map<KEY,Gdims>& v) {ddims=v; return static_cast<SPEC&>(*this);}
    
    SPEC zero() {fcode=0; return static_cast<SPEC&>(*this);}
    SPEC raw() {fcode=1; return static_cast<SPEC&>(*this);}
    SPEC ones() {fcode=2; return static_cast<SPEC&>(*this);}
    SPEC sequential() {fcode=3; return static_cast<SPEC&>(*this);}
    SPEC gaussian() {fcode=4; return static_cast<SPEC&>(*this);}
    SPEC fill(const int x) {fcode=x; return static_cast<SPEC&>(*this);}

    SPEC dev(const int i) {_dev=i; return static_cast<SPEC&>(*this);}


  public: // ---- Access ------------------------------------------------------------------------------------


    int get_nbatch() const{
      return nbatch;
    }

    Gdims get_gdims() const{
      return gdims;
    }

    map<KEY,Gdims> get_ddims() const{
      return ddims;
    }

    DimLabels get_labels() const{
      DimLabels r;
      if(nbatch>0) r._batched=true;
      r._narray=gdims.size();
      return r;
    }

    int get_fcode() const{
      return fcode;
    }

    int get_dev() const{
      return _dev;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str() const{
      ostringstream oss;
      oss<<"[ ";
      if(nbatch) oss<<"batch="<<nbatch<<" ";
      if(gdims.size()>0) oss<<"grid="<<gdims<<" ";
      if(ddims.size()>0) oss<<"dims="<<ddims<<" ";
      if(_dev>0) oss<<"dev="<<_dev<<" ";
      oss<<"]";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LtensorApackSpecBase& x){
      stream<<x.str(); return stream;
    }

  };


  template<typename KEY, typename TYPE>
  class LtensorApack;
  
  template<typename KEY, typename TYPE>
  class LtensorApackSpec: public LtensorApackSpecBase<KEY,LtensorApackSpec<KEY,TYPE> >{
  public:
    
    typedef LtensorApackSpecBase<KEY,LtensorApackSpec<KEY,TYPE> > BASE;
    using BASE::BASE;


    LtensorApack<KEY,TYPE> operator()() const{
      return LtensorApack<KEY,TYPE>(*this);
    }

  };

}

#endif 
