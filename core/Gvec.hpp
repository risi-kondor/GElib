/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GElibGvec
#define _GElibGvec

#include "GElib_base.hpp"
#include "Gpart.hpp"


namespace GElib{

  template<typename GVEC, typename GPART>
  class Gvec{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    typedef typename GPART::GROUP GROUP;
    typedef typename GPART::IRREP_IX IRREP_IX;
    typedef typename GPART::GTYPE GTYPE;

    int _nbatch=0;
    Gdims _gdims;
    int dev;

    mutable map<IRREP_IX,GPART> parts;


  public: // ---- Constructors ------------------------------------------------------------------------------


    Gvec(){}

    Gvec(const int __nbatch, const Gdims& __gdims, const int _dev=0):
      _nbatch(__nbatch),
      _gdims(__gdims),
      dev(_dev){}

    Gvec(const int __nbatch, const Gdims& __gdims, const GTYPE& tau, const int fcode=0, const int _dev=0):
      Gvec(__nbatch,__gdims,_dev){
      for(auto& p: tau.parts)
	parts.emplace(p.first,GPART(__nbatch,__gdims,p.first,p.second,fcode,_dev));
    }


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


  public: // ---- Factory methods -------------------------------------------------------------------------------------


    GVEC zeros_like() const{
      GVEC R(_nbatch,_gdims,dev);
      for(auto& p: parts)
	R.parts[p.first]=p.second.zeros_like();
    }

    template<typename GTYPE>
    GVEC zeros_like(const GTYPE& tau) const{
      GVEC R(_nbatch,_gdims,dev);
      for(auto& p: tau.map)
	R.parts.emplace(p.first,GPART(getb(),gdims(),p.first,p.second,0,get_dev()));
      return R;
    }


  public: // ---- Conversions -------------------------------------------------------------------------------


    GVEC& downcast(){
      return static_cast<GVEC&>(*this);
    }

    const GVEC& downcast() const{
      return static_cast<const GVEC&>(*this);
    }


  public: // ---- Parts -------------------------------------------------------------------------------------


    int size() const{
      return parts.size();
    }

    GTYPE get_tau() const{
      GTYPE r;
      for(auto p:parts)
	r.parts[p.first]=p.second.getn();
      return r;
    }

    bool has_part(const IRREP_IX& l) const{
      return parts.find(l)!=parts.end();
    }

    //PART operator()(const IRREP_IX& l) const{
    //auto it=parts.find(l);
    //assert(it!=parts.end());
    //return PART(*it->second);
    //}

    GPART part(const IRREP_IX& l) const{
      auto it=parts.find(l);
      assert(it!=parts.end());
      return it->second;
    }

    void for_each_part(const std::function<void(const IRREP_IX&, const GPART&)>& lambda) const{
      for(auto& p:parts) 
	lambda(p.first,p.second);
    }


  public: // ---- Access -----------------------------------------------------------------------------------


    int get_dev() const{
      return dev;
    }


  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return _nbatch>0;
    }

    int getb() const{
      return _nbatch;
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

  public: // ---- Promotions ---------------------------------------------------------------------------------


    int dominant_batch(const GVEC& y) const{
      int xb=getb();
      int yb=y.getb();
      if(xb==yb) return xb;
      if(xb==1) return yb;
      if(yb==1) return xb;
      throw std::invalid_argument("Gelib error: the batch dimensions of "+repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return 0;
    }

    Gdims dominant_gdims(const GVEC& y) const{
      Gdims xg=gdims();
      Gdims yg=y.gdims();
      if(xg==yg) return xg;
      if(!is_grid()) return yg;
      if(!y.is_grid()) return xg;
      throw std::invalid_argument("Gelib error: the grid dimensions of "+repr()+" and "+y.repr()+
	" cannot be reconciled.");
      return Gdims();
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------

    /*
    void add(const GvecE& x){
      for(auto p: parts){
	p.second->add(x.part(p.first));
      }
    }
    */

  public: // ---- Operations ---------------------------------------------------------------------------------

    
    GVEC gather(const cnine::GatherMapB& gmap, const int d=0){
      Gdims adims(_gdims);
      GELIB_ASSRT(d<adims.size());
      GELIB_ASSRT(gmap.n_in==adims(d));
      adims[d]=gmap.n_out;
      GVEC R(getb(),adims,get_dev());
      for_each_part([&](const IRREP_IX& ix , const GPART& p){
	  R.parts[ix]=p.gather(gmap,d);});
      return R;
    }


    /*
    GvecE transp(){
      GvecE r(_nbatch,_gdims,dev);
      for(auto p: parts)
	r[p.first]=p.second->transp();
      return r;
    }
    */


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    GVEC CGproduct(const GVEC& y) const{
      auto& x=static_cast<const GVEC&>(*this);
      GVEC R(x.dominant_batch(y),x.dominant_gdims(y),x.get_tau().CGproduct(y.get_tau()),0,x.get_dev());
      R.add_CGproduct(x,y);
      return R;
    }

    GVEC CGproduct(const GVEC& y, const IRREP_IX& limit) const{
      auto& x=static_cast<const GVEC&>(*this);
      //GVEC R=x.zeros_like(x.get_tau().CGproduct(y.get_tau(),limit));
      GVEC R(x.dominant_batch(y),x.dominant_gdims(y),x.get_tau().CGproduct(y.get_tau(),limit),0,x.get_dev());
      R.add_CGproduct(x,y);
      return R;
    }

    void add_CGproduct(const GVEC& x, const GVEC& y){
      GTYPE offset;
      for(auto& p:x.parts)
	for(auto& q:y.parts)
	  GROUP::for_each_CGcomponent(p.first,q.first,[&](const IRREP_IX& z, const int m){
	      if(!has_part(z)) return;
	      part(z).add_CGproduct(p.second,q.second,offset[z]);
	      offset[z]+=m*p.second.getn()*q.second.getn();
	    });
    }

    void add_CGproduct_back0(const GVEC& g, const GVEC& y){
      GTYPE offset;
      for(auto& p:parts)
	for(auto& q:y.parts)
	  GROUP::for_each_CGcomponent(p.first,q.first,[&](const IRREP_IX& z, const int m){
	      if(!g.has_part(z)) return;
	      p.second.add_CGproduct_back0(g.part(z),q.second,offset[z]);
	      offset[z]+=m*p.second.getn()*q.second.getn();
	    });
    }

    void add_CGproduct_back1(const GVEC& g, const GVEC& x){
      GTYPE offset;
      for(auto& p:x.parts)
	for(auto& q:parts)
	  GROUP::for_each_CGcomponent(p.first,q.first,[&](const IRREP_IX& z, const int m){
	      if(!g.has_part(z)) return;
	      q.second.add_CGproduct_back1(g.part(z),p.second,offset[z]);
	      offset[z]+=m*p.second.getn()*q.second.getn();
	    });
    }


  public: // ---- Diag CG-products ---------------------------------------------------------------------------

    
    GVEC DiagCGproduct(const GVEC& y, const IRREP_IX& limit=GPART::null_ix) const{
      auto& x=static_cast<const GVEC&>(*this);
      GVEC R(x.dominant_batch(y),x.dominant_gdims(y),x.get_tau().DiagCGproduct(y.get_tau(),limit),0,x.get_dev());
      R.add_DiagCGproduct(x,y);
      return R;
    }

    void add_DiagCGproduct(const GVEC& x, const GVEC& y){
      GTYPE offset;
      for(auto& p:x.parts)
	for(auto& q:y.parts)
	  GROUP::for_each_CGcomponent(p.first,q.first,[&](const IRREP_IX& z, const int m){
	      if(!has_part(z)) return;
	      part(z).add_DiagCGproduct(p.second,q.second,offset[z]);
	      offset[z]+=m*p.second.getn();
	    });
    }

    void add_DiagCGproduct_back0(const GVEC& g, const GVEC& y){
      GTYPE offset;
      for(auto& p:parts)
	for(auto& q:y.parts)
	  GROUP::for_each_CGcomponent(p.first,q.first,[&](const IRREP_IX& z, const int m){
	      if(!g.has_part(z)) return;
	      p.second.add_DiagCGproduct_back0(g.part(z),q.second,offset[z]);
	      offset[z]+=m*p.second.getn();
	    });
    }

    void add_DiagCGproduct_back1(const GVEC& g, const GVEC& x){
      GTYPE offset;
      for(auto& p:x.parts)
	for(auto& q:parts)
	  GROUP::for_each_CGcomponent(p.first,q.first,[&](const IRREP_IX& z, const int m){
	      if(!g.has_part(z)) return;
	      q.second.add_DiagCGproduct_back1(g.part(z),p.second,offset[z]);
	      offset[z]+=m*p.second.getn();
	    });
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::Gvec";
    }

    string repr() const{
      return downcast().repr();
    }
    
    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:parts){
	oss<<indent<<"Part "<<p.first<<":"<<endl;
	oss<<p.second.str(indent+"  ")<<endl;
      }
      return oss.str();
    }

    string to_print(const string indent="") const{
      ostringstream oss;
      oss<<indent<<static_cast<const GVEC&>(*this).repr()<<":"<<endl;
      oss<<str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Gvec& x){
      stream<<x.str(); return stream;
    }

  };

  
}

#endif 
