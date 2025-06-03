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

#ifndef _CnineLtensorPackSpecBase
#define _CnineLtensorPackSpecBase

#include "Cnine_base.hpp"
#include "GdimsPack.hpp"
#include "DimLabels.hpp"


namespace cnine{

  template<typename SPEC>
  class LtensorPackSpecBase{
  public:

    int nbatch=0;
    Gdims gdims;
    GdimsPack ddims;
    int _fcode=0;
    int _dev=0;


    LtensorPackSpecBase(){}

    LtensorPackSpecBase(const int _nbatch, const Gdims& _gdims, const GdimsPack& _ddims, const int __dev=0):
      nbatch(_nbatch),
      gdims(_gdims),
      ddims(_ddims),
      _dev(__dev){}


  public: // ---- Construction ------------------------------------------------------------------------------


    SPEC batch(const int b) {nbatch=b; return *this;}

    SPEC grid(const initializer_list<int>& v) {gdims=Gdims(v); return *this;}
    SPEC grid(const vector<int>& v) {gdims=Gdims(v); return *this;}
    SPEC grid(const Gdims& v) {gdims=v; return *this;}

    SPEC dims(const initializer_list<initializer_list<int> >& v) {ddims=GdimsPack(v); return *this;}
    SPEC dims(const vector<vector<int> >& v) {ddims=GdimsPack(v); return *this;}
    SPEC dims(const GdimsPack& v) {ddims=v; return *this;}
    
    SPEC zero() {_fcode=0; return *this;}
    SPEC raw() {_fcode=1; return *this;}
    SPEC ones() {_fcode=2; return *this;}
    SPEC sequential() {_fcode=3; return *this;}
    SPEC gaussian() {_fcode=4; return *this;}
    SPEC fcode(const int x) {_fcode=x; return *this;}

    SPEC dev(const int i) {_dev=i; return *this;}


  public: // ---- Access ------------------------------------------------------------------------------------


    int get_nbatch() const{
      return nbatch;
    }

    Gdims get_gdims() const{
      return gdims;
    }

    GdimsPack get_ddims() const{
      return ddims;
    }

    DimLabels get_labels() const{
      DimLabels r;
      if(nbatch>0) r._batched=true;
      r._narray=gdims.size();
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
      if(gdims.size()>0) oss<<"grid="<<gdims<<" ";
      if(ddims.size()>0) oss<<"dims="<<ddims<<" ";
      if(_dev>0) oss<<"dev="<<_dev<<" ";
      oss<<"]";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LtensorPackSpecBase<SPEC>& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
