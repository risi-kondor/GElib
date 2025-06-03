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


#ifndef _cnineLtensorPack
#define _cnineLtensorPack

#include "Ltensor.hpp"
#include "LtensorPackSpec.hpp"


namespace cnine{

  template<typename TYPE>
  class LtensorPack{
  public:

    int _nbatch=0;
    Gdims _gdims;
    int dev;
    DimLabels labels;
    vector<Ltensor<TYPE>*> parts;

    LtensorPack(){}

    ~LtensorPack(){
      for(auto& p: parts)
	delete p;
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    LtensorPack(const int __nbatch, const Gdims& __gdims,  const DimLabels& _labels, const int _dev=0):
      _nbatch(__nbatch),
      _gdims(__gdims),
      dev(_dev),
      labels(_labels){}

    //LtensorPack(const LtensorPackSpec<TYPE>& spec):
    //LtensorPack(spec.nbatch,spec.adims,spec.get_labels(),spec._dev){}

    
  public: // ---- LtensorPackSpec ---------------------------------------------------------------------------


    LtensorPack(const LtensorPackSpec<TYPE>& x):
      LtensorPack(x.get_nbatch(), x.get_gdims(), x.get_labels(), x.get_dev()){
      parts.resize(x.ddims.size());
      for(int i=0; i<x.ddims.size(); i++)
	parts[i]=new Ltensor<TYPE>(x.gdims.cat(x.ddims[i]).prepend(x.nbatch),
	  x.get_labels(),x.get_fcode(), x.get_dev()); 
    }

    static LtensorPackSpec<TYPE> make() {return LtensorPackSpec<TYPE>();}
    static LtensorPackSpec<TYPE> raw() {return LtensorPackSpec<TYPE>().raw();}
    static LtensorPackSpec<TYPE> zero() {return LtensorPackSpec<TYPE>().zero();}
    static LtensorPackSpec<TYPE> sequential() {return LtensorPackSpec<TYPE>().sequential();}
    static LtensorPackSpec<TYPE> gaussian() {return LtensorPackSpec<TYPE>().gaussian();}
    
    LtensorPackSpec<TYPE> spec() const{
      return LtensorPackSpec<TYPE>(dims,labels,dev);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    LtensorPack(const LtensorPack& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      dev(x.dev),
      labels(x.labels){
      for(auto& p:x.parts)
	parts[p.first]=new Ltensor<TYPE>(*p.second);
    }
    
    LtensorPack(LtensorPack&& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      dev(x.dev),
      labels(x.labels),
      parts(std::move(x.parts)){
    }
      
    LtensorPack& operator=(const LtensorPack& x){
      GELIB_ASSRT(_nbatch==x._nbatch);
      GELIB_ASSRT(_gdims==x._gdims);
      GELIB_ASSRT(labels==x.labels);
      for(auto& p:parts)
	(*p.second)=(*x.parts[p.first]);
      return *this;
    }

    LtensorPack copy() const{
      LtensorPack r(_nbatch,_gdims,labels,dev);
      for(auto& p:parts)
	r.parts(p.first)=new Ltensor<TYPE>(p.second->copy());
      return r;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------

    
    #ifdef _WITH_ATEN
    
    vector<at::Tensor> torch() const{
      vector<at::Tensor> R;
      for_each([&](const Ltensor<TYPE>& c){
	  R.push_back(x.torch());});
      return R;
    }

    #endif 


  public: // ---- Access ------------------------------------------------------------------------------------



  public: // ---- Parts -------------------------------------------------------------------------------------


    int size() const{
      return parts.size();
    }

    Ltensor<TYPE> operator[](const int i) const{
      CNINE_ASSRT(i<parts.size());
      return parts[i];
    }

    void for_each(const std::function<void(const int, const Ltensor<TYPE>&)>& lambda) const{
      for(int i=0; i<parts.size(); i++)
	lambda(i,*parts[i]);
    }


  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return _nbatch>0;
    }

    int nbatch() const{
      return _nbatch;
    }

    LtensorPack batch(const int b) const{
      CNINE_ASSRT(is_batched());
      CNINE_ASSRT(b>=0 && b<_nbatch);
      LtensorPack r(0,_gdims,labels.copy().set_batched(false),dev);
      r.parts.resize(size());
      for(int i=0; i<parts.size(); i++)
	r.parts[i]=new Ltensor<TYPE>(parts[i]->batch(b));
      return r;
    }

    void for_each_batch(const std::function<void(const int, const LtensorPack& x)>& lambda) const{
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

    LtensorPack cell(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==_gdims.size());
      LtensorPack r(_nbatch,cnine::Gdims(),labels.copy().set_ngrid(0),dev);
      r.parts.resize(size());
      for(int i=0; i<parts.size(); i++)
	r.parts[i]=new Ltensor<TYPE>(parts[i]->cell(ix));
      return r;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const LtensorPack& x){
      CNINE_ASSRT(x.size()==size());
      for(int i=0; i<parts.size(); i++)
	parts[i]->add(*x.parts[i]);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    LtensorPack transp(){
      LtensorPack r(_nbatch,_gdims,labels,dev);
      r.parts.resize(size());
      for(int i=0; i<parts.size(); i++)
	r.parts[i]=new Ltensor<TYPE>(parts[i]->transp());
      return r;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "LtensorPack";
    }

    string repr() const{
      ostringstream oss;
      oss<<"LtensorPack"<<"["<<dev<<"]";
      return oss.str();
    }

    string to_string(const string indent="") const{
      ostringstream oss;
      for_each([&](const int i, const Ltensor<TYPE>& x){
	  oss<<indent<<"Tensor "<<i<<":"<<endl;
	  oss<<x.str(indent)<<endl;
	});
      return oss.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<":"<<endl;
      oss<<to_string(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LtensorPack<TYPE>& x){
      stream<<x.str(); return stream;
    }
  };

}

#endif 
