// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGvecD
#define _GElibGvecD

#include "GElib_base.hpp"
#include "GvecSpec.hpp"


namespace GElib{

  template<typename GROUP, typename TYPE>
  class GvecD{
  public:

    typedef typename GROUP::IrrepIx KEY;
    //typedef typename GROUP::template PART<TYPE> PART;
    typedef decltype(GROUP::template dummy_part<TYPE>()) PART;
    typedef typename GROUP::TAU TAU;

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    int _nbatch=0;
    Gdims _gdims;
    int dev;
    //cnine::DimLabels labels;
    mutable map<KEY,PART*> parts;

    GvecD(){}

    ~GvecD(){
      for(auto& p: parts)
	delete p.second;
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    GvecD(const int __nbatch, const Gdims& __gdims, const int _dev=0):
      _nbatch(__nbatch),
      _gdims(__gdims),
      dev(_dev){}

    template<typename SPEC>
    GvecD(const GvecSpec<SPEC,TAU>& spec):
      _nbatch(spec.nbatch), 
      _gdims(spec.adims),
      dev(spec._dev) //labels(spec.get_labels())
    {}


  public: // ---- Copying -----------------------------------------------------------------------------------


    GvecD(const GvecD& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      dev(x.dev){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=new PART(*p.second);
    }
    
    GvecD(GvecD&& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      dev(x.dev),
      parts(std::move(x.parts)){
      GELIB_MOVE_WARNING();
    }
      
    GvecD& operator=(const GvecD& x){
      GELIB_ASSIGN_WARNING();
      GELIB_ASSRT(_nbatch==x._nbatch);
      GELIB_ASSRT(_gdims==x._gdims);
      for(auto& p:parts)
	(*p.second)=(*x.parts[p.first]);
      return *this;
    }

    GvecD copy() const{
      GvecD r(_nbatch,_gdims,dev);
      for(auto& p:parts)
	r.parts(p.first)=new PART(p.second->copy());
      return r;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------

    
    #ifdef _WITH_ATEN
    
    vector<at::Tensor> torch() const{
      vector<at::Tensor> R;
      for_each_part([&](const KEY& key, const PART& part){
	  R.push_back(part.torch());});
      return R;
    }

    #endif 


  public: // ---- Access ------------------------------------------------------------------------------------



  public: // ---- Parts -------------------------------------------------------------------------------------


    int size() const{
      return parts.size();
    }

    TAU tau() const{
      TAU r;
      for(auto p:parts)
	r[p.first]=p.second->getn();
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


  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return _nbatch>0;
    }

    int nbatch() const{
      return _nbatch;
    }

    GvecD batch(const int i) const{
      CNINE_ASSRT(is_batched());
      CNINE_ASSRT(i>=0 && i<_nbatch);
      //GvecD r(0,_gdims,labels.copy().set_batched(false),dev);
      GvecD r(0,_gdims,dev);
      for(auto p:parts)
	r.parts[p.first]=new PART(p.second->batch(i));
      return r;
    }

    void for_each_batch(const std::function<void(const int, const GvecD& x)>& lambda) const{
      int B=nbatch();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }


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

    GvecD cell(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==_gdims.size());
      GvecD r(_nbatch,cnine::Gdims(),dev);
      for(auto p: parts)
	r[p.first]=new PART(p.second->cell(ix));
      return r;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const GvecD& x){
      for(auto p: parts){
	p.second->add(x.part(p.first));
      }
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


    void add_CGproduct(const GvecD& x, const GvecD& y){
      TAU offs;
      for(auto p1: x.parts)
	for(auto p2: y.parts){
	  const PART& P1=*p1.second;
	  const PART& P2=*p2.second;
	  GROUP::for_each_CGcomponent(p1.first,p2.first,
	    [&](const KEY l, const int m){
	      if(has_part(l)){
		parts[l]->add_CGproduct(P1,P2,offs[l]);
		offs[l]+=P1.getn()*P2.getn()*m;
	      }
	    });
	}
    }


    void add_CGproduct_back0(const GvecD& g, const GvecD& y){
      TAU offs;
      for(auto p1: parts)
	for(auto p2: y.parts){
	  PART& P1=*p1.second;
	  const PART& P2=*p2.second;
	  GROUP::for_each_CGcomponent(p1.first,p2.first,
	    [&](const KEY l, const int m){
	      if(g.has_part(l)){
		P1.add_CGproduct_back0(*g.parts[l],P2,offs[l]);
		offs[l]+=P1.getn()*P2.getn()*m;
	      }
	    });
	}
    }


    void add_CGproduct_back1(const GvecD& g, const GvecD& x){
      TAU offs;
      for(auto p1: x.parts)
	for(auto p2: parts){
	  const PART& P1=*p1.second;
	  PART& P2=*p2.second;
	  GROUP::for_each_CGcomponent(p1.first,p2.first,
	    [&](const KEY l, const int m){
	      if(g.has_part(l)){
		P2.add_CGproduct_back1(*g.parts[l],P1,offs[l]);
		offs[l]+=P1.getn()*P2.getn()*m;
	      }
	    });
	}
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::GvecD";
    }

    string repr() const{
      ostringstream oss;
      //oss<<"GvecD(";
      oss<<GROUP::Gname()<<"vecD(";
      if(is_batched()) oss<<"b="<<nbatch()<<",";
      if(is_grid()) oss<<"grid="<<gdims()<<",";
      oss<<"tau="<<tau()<<",";
      if(dev>0) oss<<"dev="<<dev<<",";
      oss<<"\b)";
      //"<<"["<<dev<<"]";
      return oss.str();
    }
    
    string str(const string indent="", bool norepr=false) const{
      ostringstream oss;
      if(!norepr) oss<<indent<<repr()<<":"<<endl;
      if(is_batched()){
	for_each_batch([&](const int b, const GvecD& x){
	    oss<<indent<<"  "<<"Batch "<<b<<":"<<endl;
	    oss<<x.str(indent+"  ",true);
	  });
      }else{
	for_each_part([&](const KEY p, const PART& x){
	    oss<<indent<<"  "<<"Part "<<p<<":"<<endl;
	    oss<<x.str(indent+"  ");
	  });
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GvecD& x){
      stream<<x.str(); return stream;
    }

  };

  

  
}

#endif 
    //GvecD(const int __nbatch, const Gdims& __gdims, const cnine::DimLabels& _labels, const int _dev=0):
    //_nbatch(__nbatch),
    //_gdims(__gdims),
    //labels(_labels),
    //dev(_dev){
    //}

