
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3vecB
#define _SO3vecB

#include "GElib_base.hpp"
#include "SO3type.hpp"
#include "SO3partB.hpp"
#include "SO3element.hpp"


namespace GElib{


  class SO3vecB{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorPackObj ctensorpack;


    vector<SO3partB*> parts;


    SO3vecB(){}

    ~SO3vecB(){
      for(auto p: parts) delete p;  
    }


    // ---- Constructors --------------------------------------------------------------------------------------


    //SO3vecB(const cnine::fill_noalloc& dummy, const SO3type& _tau, const int _nbu, const int _fmt, const int _dev):
    //tau(_tau), nbu(_nbu), fmt(_fmt), dev(_dev){}


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecB(const int b, const SO3type& tau, const FILLTYPE fill, const int _dev){
      for(int l=0; l<tau.size(); l++)
	parts.push_back(new SO3partB(b,l,tau[l],fill,_dev));
    }

    
    // ---- Named constructors --------------------------------------------------------------------------------

    
    static SO3vecB zero(const int b, const SO3type& tau, const int _dev=0){
      return SO3vecB(b,tau,cnine::fill_zero(),_dev);
    }
    
    static SO3vecB gaussian(const int b, const SO3type& tau, const int _dev=0){
      return SO3vecB(b,tau,cnine::fill_gaussian(),_dev);
    }
    
    
    // ---- Copying -------------------------------------------------------------------------------------------


    SO3vecB(const SO3vecB& x){
      for(auto& p:x.parts)
	parts.push_back(p);
    }

    SO3vecB(SO3vecB&& x){
      parts=x.parts;
      x.parts.clear();
    }


    // ---- Transport -----------------------------------------------------------------------------------------


    SO3vecB& move_to_device(const int _dev){
      for(auto p:parts)
	p->move_to_device(_dev);
      return *this;
    }
    
    SO3vecB to_device(const int _dev) const{
      SO3vecB R;
      for(auto p:parts)
	R.parts.push_back(new SO3partB(p->to_device(_dev)));
      return R;
    }


  public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    SO3vecB(vector<at::Tensor>& v){
      for(auto& p: v)
	parts.push_back(new SO3partB(p));
    }

#endif

 
    // ---- Access --------------------------------------------------------------------------------------------


    int getb() const{
      if(parts.size()>0) return parts[0]->getb();
      return 0;
    }

    SO3type get_tau() const{
      SO3type tau;
      for(auto p:parts)
	tau.push_back(p->getn());
      return tau;
    }

    int get_maxl() const{
      return parts.size()-1;
    }

    int get_dev() const{
      if(parts.size()>0) return parts[0]->get_dev();
      return 0;
    }

    int get_device() const{
      if(parts.size()>0) return parts[0]->get_dev();
      return 0;
    }

    

    // ---- Rotations ----------------------------------------------------------------------------------------


    SO3vecB rotate(const SO3element& r){
      SO3vecB R;
      for(int l=0; l<parts.size(); l++)
	if(parts[l]) R.parts.push_back(new SO3partB(parts[l]->rotate(r)));
	else R.parts.push_back(nullptr);
      return R;
    }

    
    // ---- CG-products ---------------------------------------------------------------------------------------


    SO3vecB CGproduct(const SO3vecB& y, const int maxl=-1) const{
      assert(getb()==y.getb());

      SO3vecB R=SO3vecB::zero(getb(),GElib::CGproduct(get_tau(),y.get_tau(),maxl),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }


    void add_CGproduct(const SO3vecB& x, const SO3vecB& y){
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
	    parts[l]->add_CGproduct(*x.parts[l1],*y.parts[l2],offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*(y.parts[l2]->getn());
	  }
	}
      }
    }

      
    void add_CGproduct_back0(const SO3vecB& g, const SO3vecB& y){
      assert(g.get_tau()==GElib::CGproduct(get_tau(),y.get_tau(),g.get_maxl()));

      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
      vector<int> offs(parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l1]->add_CGproduct_back0(*g.parts[l],*y.parts[l2],offs[l]);
	    offs[l]+=(parts[l1]->getn())*(y.parts[l2]->getn());
	  }
	}
      }
    }

      
    void add_CGproduct_back1(const SO3vecB& g, const SO3vecB& x){
      assert(g.get_tau()==GElib::CGproduct(x.get_tau(),get_tau(),g.get_maxl()));

      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=g.get_maxl();
      vector<int> offs(parts.size(),0);
	
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


    string str(const string indent="") const{
      ostringstream oss;
	for(int l=0; l<parts.size(); l++){
	  if(!parts[l]) continue;
	  oss<<indent<<"Part l="<<l<<":\n";
	  oss<<parts[l]->str(indent+"  ");
	  oss<<endl;
	}
      return oss.str();
    }

    string repr(const string indent="") const{
      return "<GElib::SO3vecB of type"+get_tau().str()+">";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3vecB& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif
