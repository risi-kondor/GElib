// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGvec
#define _GElibGvec

#include "GElib_base.hpp"
#include "Gpart.hpp"


namespace GElib{

  template<typename PART>
  class Gvec{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;
    typedef typename PART::IrrepIx IrrepIx;

    int _nbatch=0;
    Gdims _gdims;
    int dev;

    mutable map<IrrepIx,PART> parts;


  public: // ---- Constructors ------------------------------------------------------------------------------


  public: // ---- Named parameter constructors ---------------------------------------------------------------


    struct vparams{
      int b=1;
      Gdims gdims;
      int nc=1;
      std::any tau;
      int fcode=0;
      int dev=0;
    };      

    template<typename... Args>
    void unroller(vparams& v, const cnine::BatchArgument& x, const Args&... args){
      v.b=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::GridArgument& x, const Args&... args){
      v.gdims=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const TtypeArgument& x, const Args&... args){
      v.tau=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::FillArgument& x, const Args&... args){
      v.fcode=x.get(); unroller(v, args...);}

    template<typename... Args>
    void unroller(vparams& v, const cnine::DeviceArgument& x, const Args&... args){
      v.dev=x.get(); unroller(v, args...);}

    void unroller(vparams& v){}

    void reset(const vparams& v){
      _nbatch=v.b;
      _gdims=v.gdims;
      dev=v.dev;
    }


  public: // ---- Parts -------------------------------------------------------------------------------------


    int size() const{
      return parts.size();
    }

    /*
    GtypeE tau() const{
      GtypeE r;
      for(auto p:parts)
	r.map[p.first]=p.second->getn();
      return r;
    }

    bool has_part(const KEY& l) const{
      return parts.find(l)!=parts.end();
    }

    PART operator()(const KEY& l) const{
      auto it=parts.find(l);
      assert(it!=parts.end());
      return PART(*it->second);
    }

    PART part(const KEY& l) const{
      auto it=parts.find(l);
      assert(it!=parts.end());
      return PART(*it->second);
    }

    void for_each_part(const std::function<void(const KEY&, const PART&)>& lambda) const{
      for(auto& p:parts) 
	lambda(p.first,*p.second);
    }
    */

  public: // ---- Access -----------------------------------------------------------------------------------


    int get_dev() const{
      return dev;
    }


  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return _nbatch>0;
    }

    int nbatch() const{
      return _nbatch;
    }

    /*
    GvecE batch(const int i) const{
      CNINE_ASSRT(is_batched());
      CNINE_ASSRT(i>=0 && i<_nbatch);
      //GvecE r(0,_gdims,labels.copy().set_batched(false),dev);
      GvecE r(0,_gdims,dev);
      for(auto p:parts)
	r.parts[p.first]=new PART(p.second->batch(i));
      return r;
    }

    void for_each_batch(const std::function<void(const int, const GvecE& x)>& lambda) const{
      int B=nbatch();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }
    */

  public: // ---- Grid ---------------------------------------------------------------------------------------


    bool is_grid() const{
      return _gdims.size()>0;
    }

    int ngdims() const{
      return _gdims.size();
    }

    Gdims gdims() const{
      return _gdims;
    }

    /*
    GvecE cell(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==_gdims.size());
      GvecE r(_nbatch,cnine::Gdims(),dev);
      for(auto p: parts)
	r[p.first]=new PART(p.second->cell(ix));
      return r;
    }
    */

  public: // ---- Cumulative operations ----------------------------------------------------------------------

    /*
    void add(const GvecE& x){
      for(auto p: parts){
	p.second->add(x.part(p.first));
      }
    }
    */

  public: // ---- Operations ---------------------------------------------------------------------------------


    /*
    GvecE transp(){
      GvecE r(_nbatch,_gdims,dev);
      for(auto p: parts)
	r[p.first]=p.second->transp();
      return r;
    }
    */


    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::Gvec";
    }

    string repr() const{
      ostringstream oss;
      return oss.str();
    }
    
    string str(const string indent="", bool norepr=false) const{
      ostringstream oss;
      if(!norepr) oss<<indent<<repr()<<":"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Gvec& x){
      stream<<x.str(); return stream;
    }

  };

  
}

#endif 
