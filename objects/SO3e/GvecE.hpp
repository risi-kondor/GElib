// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGvecE
#define _GElibGvecE

#include "GElib_base.hpp"
#include "GvecSpec.hpp"


namespace GElib{

  template<typename TYPE>
  class GvecE{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;

    shared_ptr<Ggroup> G;
    int _nbatch=0;
    Gdims _gdims;
    int dev;
    mutable map<GirrepIxWrapper,GpartE<TYPE>*> parts;

    GvecE(shared_ptr<Ggroup>& _G): 
      G(_G){}

    ~GvecE(){
      for(auto& p: parts)
	delete p.second;
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    GvecE(shared_ptr<Ggroup>& _G, const int __nbatch, const Gdims& __gdims, const int _dev=0):
      G(_G),
      _nbatch(__nbatch),
      _gdims(__gdims),
      dev(_dev){}

    GvecE(const GvecSpec& spec):
      G(spec.G),
      _nbatch(spec.nbatch), 
      _gdims(spec.adims),
      dev(spec._dev) //labels(spec.get_labels())
    {}


  public: // ---- Copying -----------------------------------------------------------------------------------


    GvecE(const GvecE& x):
      G(x.G),
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      dev(x.dev){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts[p.first]=new GpartE<TYPE>(*p.second);
    }
    
    GvecE(GvecE&& x):
      G(x.G),
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      dev(x.dev),
      parts(std::move(x.parts)){
      GELIB_MOVE_WARNING();
    }
      
    GvecE& operator=(const GvecE& x){
      GELIB_ASSIGN_WARNING();
      GELIB_ASSRT(_nbatch==x._nbatch);
      GELIB_ASSRT(_gdims==x._gdims);
      G.reset(x.G);
      for(auto& p:parts)
	(*p.second)=(*x.parts[p.first]);
      return *this;
    }

    GvecE copy() const{
      GvecE r(G,_nbatch,_gdims,dev);
      for(auto& p:parts)
	r.parts(p.first)=new GpartE<TYPE>(p.second->copy());
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

    GtypeE tau() const{
      GtypeE r;
      for(auto p:parts)
	r.map[p.first]=p.second->getn();
      return r;
    }

    /*
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
      return "GElib::GvecE";
    }

    string repr() const{
      ostringstream oss;
      //oss<<"GvecE(";
      oss<<"GvecE("<<G->repr()<<",";
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
      /*
     if(is_batched()){
	for_each_batch([&](const int b, const GvecE& x){
	    oss<<indent<<"  "<<"Batch "<<b<<":"<<endl;
	    oss<<x.str(indent+"  ",true);
	  });
      }else{
	for_each_part([&](const KEY p, const PART& x){
	    oss<<indent<<"  "<<"Part "<<p<<":"<<endl;
	    oss<<x.str(indent+"  ");
	  });
      }
      */
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GvecE& x){
      stream<<x.str(); return stream;
    }

  };

  
}

#endif 
