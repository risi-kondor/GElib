
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3vecB_array
#define _SO3vecB_array

#include "GElib_base.hpp"
#include "SO3type.hpp"
#include "CtensorB_multiArray.hpp"
#include "SO3partB_array.hpp"
#include "SO3element.hpp"


namespace GElib{

  //template<typename ARRAY>
  typedef cnine::CtensorB_multiArray<SO3partB_array> SO3vecB_base;


  class SO3vecB_array: public cnine::CtensorB_multiArray<SO3partB_array>{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;
    typedef cnine::Gdims Gdims;

    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorPackObj ctensorpack;

    //using CtensorB_multiArray<SO3partB_array>::CtensorB_multiArray<SO3partB_array>;
    using SO3vecB_base::SO3vecB_base;

    //vector<SO3partB_array*> parts;


    SO3vecB_array(){}

    ~SO3vecB_array(){
      //for(auto p: parts) delete p;  
    }


    // ---- Constructors --------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecB_array(const Gdims& _adims, const SO3type& tau, const FILLTYPE fill, const int _dev){
      for(int l=0; l<tau.size(); l++)
	parts.push_back(new SO3partB_array(_adims,l,tau[l],fill,_dev));
    }

    
    // ---- Named constructors --------------------------------------------------------------------------------

    
    static SO3vecB_array zero(const Gdims& _adims, const SO3type& tau, const int _dev=0){
      return SO3vecB_array(_adims,tau,cnine::fill_zero(),_dev);
    }
  
    static SO3vecB_array gaussian(const Gdims& _adims, const SO3type& tau, const int _dev=0){
      return SO3vecB_array(_adims,tau,cnine::fill_gaussian(),_dev);
    }

    static SO3vecB_array zeros_like(const SO3vecB_array& x){
      return SO3vecB_array::zero(x.get_adims(),x.get_tau(),x.get_dev());
    }

    static SO3vecB_array gaussian_like(const SO3vecB_array& x){
      return SO3vecB_array::gaussian(x.get_adims(),x.get_tau(),x.get_dev());
    }


  public: // ---- Copying -------------------------------------------------------------------------------------------


    SO3vecB_array(const SO3vecB_array& x){
      for(auto& p:x.parts)
	parts.push_back(new SO3partB_array(*p));
    }

    SO3vecB_array(SO3vecB_array&& x){
      parts=x.parts;
      x.parts.clear();
    }


  public: // ---- Transport -----------------------------------------------------------------------------------------


    //SO3vecB_array& move_to_device(const int _dev){
    //for(auto p:parts)
    //p->move_to_device(_dev);
    //return *this;
    //}
    
    //SO3vecB_array to_device(const int _dev) const{
    //SO3vecB_array R;
    //for(auto p:parts)
    //R.parts.push_back(new SO3partB_array(p->to_device(_dev)));
    //return R;
    //}


  public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    SO3vecB_array(vector<at::Tensor>& v){
      for(auto& p: v)
	parts.push_back(new SO3partB_array(p));
    }

#endif

  
  public: // ---- Access --------------------------------------------------------------------------------------------
  

    //Gdims get_adims() const{
    //if(parts.size()>0) return parts[0]->get_adims();
    //return 0;
    //}

    SO3type get_tau() const{
      SO3type tau;
      for(auto p:parts)
	tau.push_back(p->getn());
      return tau;
    }

    int get_maxl() const{
      return parts.size()-1;
    }

    //int get_dev() const{
    //if(parts.size()>0) return parts[0]->get_dev();
    //return 0;
    //}

    //int get_device() const{
    //if(parts.size()>0) return parts[0]->get_dev();
    //return 0;
    //}

    

  public: // ---- Operations ---------------------------------------------------------------------------------------


    //SO3vecB_array operator-(const SO3vecB_array& y) const{
    //SO3vecB_array R;
    //for(int l=0; l<parts.size(); l++){
    //R.parts.push_back(new SO3partB_array((*parts[l])-(*y.parts[l])));
    //}
    //return R;
    //}



  public: // ---- Rotations ----------------------------------------------------------------------------------------


    SO3vecB_array rotate(const SO3element& r){
      SO3vecB_array R;
      for(int l=0; l<parts.size(); l++)
	if(parts[l]) R.parts.push_back(new SO3partB_array(parts[l]->rotate(r)));
	else R.parts.push_back(nullptr);
      return R;
    }

    
  public: // ---- Cumulative Operations -----------------------------------------------------------------------------


    //void add_gather(const SO3vecB_array& x, const cnine::Rmask1& mask){
    //assert(parts.size()==x.parts.size());
    //for(int l=0; l<parts.size(); l++)
    //parts[l]->add_gather(*x.parts[l],mask);
    //}
    
    
  public: // ---- CG-products ---------------------------------------------------------------------------------------


    SO3vecB_array CGproduct(const SO3vecB_array& y, const int maxl=-1) const{
      assert(get_adims()==y.get_adims());
      SO3vecB_array R=SO3vecB_array::zero(get_adims(),GElib::CGproduct(get_tau(),y.get_tau(),maxl),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }


    void add_CGproduct(const SO3vecB_array& x, const SO3vecB_array& y){
      assert(get_tau()==GElib::CGproduct(x.get_tau(),y.get_tau(),get_maxl()));

      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
      vector<int> offs(parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	//if(x.tau[l1]==0) continue;
	for(int l2=0; l2<=L2; l2++){
	  //if(y.tau[l2]==0) continue;
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    //cout<<l1<<l2<<l<<endl;
      //cout<<parts.size()<<endl;
      //cout<<*parts[l]<<endl;
	    parts[l]->add_CGproduct(*x.parts[l1],*y.parts[l2],offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*(y.parts[l2]->getn());
	  }
	}
      }
    }

      
    void add_CGproduct_back0(const SO3vecB_array& g, const SO3vecB_array& y){
      assert(g.get_tau()==GElib::CGproduct(get_tau(),y.get_tau(),g.get_maxl()));

      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
      vector<int> offs(g.parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l1]->add_CGproduct_back0(*g.parts[l],*y.parts[l2],offs[l]);
	    offs[l]+=(parts[l1]->getn())*(y.parts[l2]->getn());
	  }
	}
      }
    }

      
    void add_CGproduct_back1(const SO3vecB_array& g, const SO3vecB_array& x){
      assert(g.get_tau()==GElib::CGproduct(x.get_tau(),get_tau(),g.get_maxl()));

      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=g.get_maxl();
      vector<int> offs(g.parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l2]->add_CGproduct_back1(*g.parts[l],*x.parts[l1],offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*(parts[l2]->getn());
	  }
	}
      }
    }

      
  public: // ---- I/O ---------------------------------------------------------------------------------------


    //string str(const string indent="") const{
    //ostringstream oss;
    //for(int l=0; l<parts.size(); l++){
    //  if(!parts[l]) continue;
    //  oss<<indent<<"Part l="<<l<<":\n";
    //  oss<<parts[l]->str(indent+"  ");
    //  oss<<endl;
    //}
    //return oss.str();
    //}

    string repr(const string indent="") const{
      return "<GElib::SO3vecB_array of type("+get_adims().str()+","+get_tau().str()+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3vecB_array& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif

