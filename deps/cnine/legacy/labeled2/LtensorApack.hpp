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


#ifndef _cnineLtensorApack
#define _cnineLtensorApack

#include "Ltensor.hpp"
#include "LtensorApackSpec.hpp"


namespace cnine{


  template<typename KEY, typename TYPE>
  class LtensorApack{
  public:

    int _nbatch=0;
    Gdims _gdims;
    int _dev;
    DimLabels _labels;
    map<KEY,Ltensor<TYPE> > tensors;

    LtensorApack(){}

    ~LtensorApack(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    LtensorApack(const int __nbatch, const Gdims& __gdims,  const DimLabels& __labels, const int __dev=0):
      _nbatch(__nbatch),
      _gdims(__gdims),
      _dev(__dev),
      _labels(__labels){}

    
  public: // ---- LtensorApackSpec ---------------------------------------------------------------------------


    LtensorApack(const LtensorApackSpec<KEY,TYPE>& x):
      LtensorApack(x.get_nbatch(), x.get_gdims(), x.get_labels(), x.get_dev()){
      int fcode=x.get_fcode();
      for(auto& ddim: x.ddims)
	tensors.emplace(ddim.first,Ltensor<TYPE>(x.gdims.cat(ddim.second).prepend(x.nbatch),_labels,fcode,_dev)); 
    }

    static LtensorApackSpec<KEY,TYPE> make() {return LtensorApackSpec<KEY,TYPE>();}
    static LtensorApackSpec<KEY,TYPE> raw() {return LtensorApackSpec<KEY,TYPE>().raw();}
    static LtensorApackSpec<KEY,TYPE> zero() {return LtensorApackSpec<KEY,TYPE>().zero();}
    static LtensorApackSpec<KEY,TYPE> sequential() {return LtensorApackSpec<KEY,TYPE>().sequential();}
    static LtensorApackSpec<KEY,TYPE> gaussian() {return LtensorApackSpec<KEY,TYPE>().gaussian();}
    
    //LtensorApackSpec<KEY,TYPE> spec() const{
    //return LtensorApackSpec<KEY,TYPE>(dims,labels,dev);
    //}


  public: // ---- Copying -----------------------------------------------------------------------------------


    LtensorApack(const LtensorApack& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      _dev(x._dev),
      _labels(x._labels){
      for(auto& p:x.tensors)
	tensors[p.first]=Ltensor<TYPE>(p.second);
    }
    
    LtensorApack(LtensorApack&& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      _dev(x._dev),
      _labels(x._labels),
      tensors(std::move(x.tensors)){
    }
      
    LtensorApack& operator=(const LtensorApack& x){
      GELIB_ASSRT(_nbatch==x._nbatch);
      GELIB_ASSRT(_gdims==x._gdims);
      GELIB_ASSRT(_labels==x._labels);
      _dev=x._dev;
      for(auto& p:tensors)
	p.second=x.tensors[p.first];
      return *this;
    }

    LtensorApack copy() const{
      LtensorApack r(_nbatch,_gdims,_labels,_dev);
      for(auto& p:tensors)
	r.tensors.emplace(p.first,p.second->copy());
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



  public: // ---- Tensors -------------------------------------------------------------------------------------


    int size() const{
      return tensors.size();
    }

    Ltensor<TYPE> operator[](const KEY& x) const{
      CNINE_ASSRT(tensors.find(x)!=tensors.end());
      return const_cast<LtensorApack&>(*this).tensors[x];
    }

    Ltensor<TYPE>& operator[](const KEY& x){
      CNINE_ASSRT(tensors.find(x)!=tensors.end());
      return const_cast<LtensorApack&>(*this).tensors[x];
    }

    void for_each(const std::function<void(const KEY&, const Ltensor<TYPE>&)>& lambda) const{
      for(auto p: tensors)
	lambda(p.first,p.second);
    }

    LtensorApack mapcar(const std::function<Ltensor<TYPE>(const KEY&, const Ltensor<TYPE>& )>& lambda){
      LtensorApack r(_nbatch,_gdims,_labels,_dev);
      for(auto p: tensors)
	r.emplace(p.first,lambda(p.first,p.second));
      return r;
    }



  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return _nbatch>0;
    }

    int nbatch() const{
      return _nbatch;
    }

    LtensorApack batch(const int b) const{
      CNINE_ASSRT(is_batched());
      CNINE_ASSRT(b>=0 && b<_nbatch);
      LtensorApack r(0,_gdims,_labels.copy().set_batched(false),_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,p.second.batch(b));
      return r;
    }

    void for_each_batch(const std::function<void(const int, const LtensorApack& x)>& lambda) const{
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

    LtensorApack cell(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==_gdims.size());
      LtensorApack r(_nbatch,cnine::Gdims(),_labels.copy().set_ngrid(0),_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,p.second.cell(ix));
      return r;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const LtensorApack& x){
      CNINE_ASSRT(x.size()==size());
      for(auto p:tensors)
	p.second.add(x.tensors[p.first]);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    LtensorApack transp(){
      LtensorApack r(_nbatch,_gdims,_labels,_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,p.second.transp());
      return r;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "LtensorApack";
    }

    string repr() const{
      ostringstream oss;
      oss<<"LtensorApack(";
      if(is_batched()) oss<<"b="<<nbatch()<<",";
      if(is_grid()) oss<<"grid="<<gdims()<<",";
      if(_dev>0) oss<<"dev="<<_dev<<",";
      if(is_batched()||is_grid()||_dev>0) oss<<"\b";
      oss<<")";
      return oss.str();
    }

    string to_string(const string indent="") const{
      ostringstream oss;
      for_each([&](const KEY& key, const Ltensor<TYPE>& x){
	  oss<<indent<<"Tensor "<<key<<":"<<endl;
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

    friend ostream& operator<<(ostream& stream, const LtensorApack& x){
      stream<<x.str(); return stream;
    }
  };

}

#endif 
