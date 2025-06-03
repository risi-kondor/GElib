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


#ifndef _cnineLtensorBpack
#define _cnineLtensorBpack

#include "Ltensor.hpp"
//#include "LtensorApackSpec.hpp"


namespace cnine{


  template<typename KEY, typename TENSOR>
  class LtensorBpack{
  public:

    int _nbatch=0;
    Gdims _gdims;
    int _dev=0;
    //DimLabels _labels;
    map<KEY,TENSOR> tensors;

    LtensorBpack(){}

    ~LtensorBpack(){
    }


  public: // ---- Constructors ------------------------------------------------------------------------------


    LtensorBpack(const int __nbatch, const Gdims& __gdims,  const int __dev=0):
      _nbatch(__nbatch),
      _gdims(__gdims),
      _dev(__dev){}

    //LtensorBpack(const int __nbatch, const Gdims& __gdims,  const DimLabels& __labels, const int __dev=0):
      //_nbatch(__nbatch),
      //_gdims(__gdims),
      //_dev(__dev),
      //_labels(__labels){}

    
  public: // ---- Copying -----------------------------------------------------------------------------------


    LtensorBpack(const LtensorBpack& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      //_labels(x._labels),
      _dev(x._dev){
      for(auto& p:x.tensors)
	tensors[p.first]=TENSOR(p.second);
    }
    
    LtensorBpack(LtensorBpack&& x):
      _nbatch(x._nbatch),
      _gdims(x._gdims),
      _dev(x._dev),
      //_labels(x._labels),
      tensors(std::move(x.tensors)){
    }
      
    LtensorBpack& operator=(const LtensorBpack& x){
      GELIB_ASSRT(_nbatch==x._nbatch);
      GELIB_ASSRT(_gdims==x._gdims);
      _dev=x._dev;
      for(auto& p:tensors)
	p.second=x.tensors[p.first];
      return *this;
    }

    LtensorBpack copy() const{
      LtensorBpack r(_nbatch,_gdims,_dev);
      for(auto& p:tensors)
	r.tensors[p.first]=p.second->copy();
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

    TENSOR operator[](const KEY& x) const{
      CNINE_ASSRT(tensors.find(x)!=tensors.end());
      return const_cast<LtensorBpack&>(*this).tensors[x];
    }

    void for_each(const std::function<void(const KEY&, const TENSOR&)>& lambda) const{
      for(auto p: tensors)
	lambda(p.first,p.second);
    }

    LtensorBpack mapcar(const std::function<TENSOR(const KEY&, const TENSOR& )>& lambda){
      LtensorBpack r(_nbatch,_gdims,_dev);
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

    LtensorBpack batch(const int b) const{
      CNINE_ASSRT(is_batched());
      CNINE_ASSRT(b>=0 && b<_nbatch);
      LtensorBpack r(0,_gdims,_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,p.second.batch(b));
      return r;
    }

    void for_each_batch(const std::function<void(const int, const LtensorBpack& x)>& lambda) const{
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

    LtensorBpack cell(const Gindex& ix) const{
      CNINE_ASSRT(ix.size()==_gdims.size());
      LtensorBpack r(_nbatch,cnine::Gdims(),_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,p.second.cell(ix));
      return r;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const LtensorBpack& x){
      CNINE_ASSRT(x.size()==size());
      for(auto p:tensors)
	p.second.add(x.tensors[p.first]);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    LtensorBpack transp(){
      LtensorBpack r(_nbatch,_gdims,_dev);
      for(auto p:tensors)
	r.tensors.emplace(p.first,p.second.transp());
      return r;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "LtensorBpack";
    }

    string repr() const{
      ostringstream oss;
      oss<<"LtensorBpack(";
      if(is_batched()) oss<<"b="<<nbatch()<<",";
      if(is_grid()) oss<<"grid="<<gdims()<<",";
      if(_dev>0) oss<<"dev="<<_dev<<",";
      if(is_batched()||is_grid()||_dev>0) oss<<"\b";
      oss<<")";
      return oss.str();
    }

    string to_string(const string indent="") const{
      ostringstream oss;
      for_each([&](const KEY& key, const TENSOR& x){
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

    friend ostream& operator<<(ostream& stream, const LtensorBpack& x){
      stream<<x.str(); return stream;
    }
  };

}

#endif 
