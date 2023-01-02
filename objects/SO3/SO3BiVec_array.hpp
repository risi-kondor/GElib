
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3BiVec_array
#define _SO3BiVec_array

#include "GElib_base.hpp"
#include "SO3BiType.hpp"
#include "MultiTensorArray.hpp"
#include "SO3BiPart_array.hpp"
//#include "SO3partB_array.hpp"
#include "SO3element.hpp"

namespace GElib{

  typedef cnine::MultiTensorArray<pair<int,int>,SO3BiPart_array> MTA;


  class SO3BiVec_array: public cnine::MultiTensorArray<pair<int,int>,SO3BiPart_array>{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;
    typedef cnine::Gdims Gdims;

    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorPackObj ctensorpack;

    //using SO3vecB_base::SO3vecB_base;

    //vector<SO3BiPart_array*> parts;


    SO3BiVec_array(){}

    ~SO3BiVec_array(){
      //for(auto p: parts) delete p;  
    }


    // ---- Constructors --------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3BiVec_array(const Gdims& _adims, const SO3BiType& tau, const FILLTYPE fill, const int _dev){
      tau.for_each([&](const int l1, const int l2, const int m){
	  insert(l1,l2,new SO3BiPart_array(_adims,l1,l2,m,fill,_dev));});
    }
	
    
    // ---- Named constructors --------------------------------------------------------------------------------


    static SO3BiVec_array zero(const Gdims& _adims, const SO3BiType& tau, const int _dev=0){
      return SO3BiVec_array(_adims,tau,cnine::fill_zero(),_dev);
    }
  
    static SO3BiVec_array gaussian(const Gdims& _adims, const SO3BiType& tau, const int _dev=0){
      return SO3BiVec_array(_adims,tau,cnine::fill_gaussian(),_dev);
    }

    static SO3BiVec_array zeros_like(const SO3BiVec_array& x){
      return SO3BiVec_array::zero(x.get_adims(),x.get_tau(),x.get_dev());
    }

    static SO3BiVec_array gaussian_like(const SO3BiVec_array& x){
      return SO3BiVec_array::gaussian(x.get_adims(),x.get_tau(),x.get_dev());
    }


  public: // ---- Copying -------------------------------------------------------------------------------------------


    SO3BiVec_array(const SO3BiVec_array& x):
      MTA(x){
      GELIB_COPY_WARNING();
    }

    SO3BiVec_array(SO3BiVec_array&& x):
      MTA(std::move(x)){
      GELIB_MOVE_WARNING();
    }

    SO3BiVec_array& operator=(const SO3BiVec_array& x){
      MTA::operator=(x);
      return *this;
    }

    SO3BiVec_array& operator=(SO3BiVec_array&& x){
      MTA::operator=(std::move(x));
      return *this;
    }


  public: // ---- Conversions ------------------------------------------------------------------------------------


    SO3BiVec_array(const MTA& x):
      MTA(x){
      GELIB_CONVERT_WARNING(x);
    }

    SO3BiVec_array(MTA&& x):
      MTA(std::move(x)){
      GELIB_MCONVERT_WARNING(x);
    }


  public: // ---- Views -------------------------------------------------------------------------------------------


    SO3BiVec_array view(){
      return SO3BiVec_array(MTA::view());
    }


  public: // ---- Transport -----------------------------------------------------------------------------------------


  public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    SO3BiVec_array(vector<at::Tensor>& v){
      for(auto& p: v)
	parts.push_back(new SO3BiPart_array(p));
    }

    vector<at::Tensor> torch(){
      vector<at::Tensor> R;
      for(auto p: parts)
	R.push_back(p->torch());
      return R;
    }

#endif

  
  public: // ---- Access --------------------------------------------------------------------------------------------
  

    Gdims get_adims() const{
      if(parts.size()>0) return parts.begin()->second->get_adims();
      return 0;
    }

    SO3BiType get_tau() const{
      SO3BiType tau;
      for(auto& p:parts)
	tau.set(p.first.first,p.first.second,p.second->getn());
      return tau;
    }

    void insert(const int l1, const int l2, SO3BiPart_array* x){
      parts[make_pair(l1,l2)]=x;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------------


    //SO3BiVec_array operator-(const SO3BiVec_array& y) const{
    //SO3BiVec_array R;
    //for(int l=0; l<parts.size(); l++){
    //R.parts.push_back(new SO3BiPart_array((*parts[l])-(*y.parts[l])));
    //}
    //return R;
    //}



  public: // ---- Rotations ----------------------------------------------------------------------------------------

    /*
    SO3BiVec_array rotate(const SO3element& r){
      SO3BiVec_array R;
      for(int l=0; l<parts.size(); l++)
	if(parts[l]) R.parts.push_back(new SO3BiPart_array(parts[l]->rotate(r)));
	else R.parts.push_back(nullptr);
      return R;
    }
    */
    
  public: // ---- Cumulative Operations -----------------------------------------------------------------------------


    //void add_gather(const SO3BiVec_array& x, const cnine::Rmask1& mask){
    //assert(parts.size()==x.parts.size());
    //for(int l=0; l<parts.size(); l++)
    //parts[l]->add_gather(*x.parts[l],mask);
    //}
    

  public: // ---- CG reduction -------------------------------------------------------------------------------


    void add_CGreduction_to(SO3vecB_array& r){
      GELIB_ASSRT(get_adims()==r.get_adims());
      SO3type offs;
      for(auto p:parts){
      }
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:parts)
	oss<<p.second->str(indent);
      return oss.str();
    }

    static string classname(){
      return "GElib::SO3BiVec_array";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3BiVec_array of type("+get_adims().str()+","+get_tau().str()+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3BiVec_array& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif

