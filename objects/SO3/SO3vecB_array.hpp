
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
#include "GElibTimer.hpp"

namespace GElib{

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

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecB_array(const Gdims& _adims, const int maxl, const FILLTYPE fill, const int _dev=0){
      for(int l=0; l<=maxl; l++)
	parts.push_back(new SO3partB_array(_adims,l,2*l+1,fill,_dev));
    }

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecB_array(const int b, const Gdims& _adims, const SO3type& tau, const FILLTYPE fill, const int _dev){
      for(int l=0; l<tau.size(); l++){
	parts.push_back(new SO3partB_array(b,_adims,l,tau[l],fill,_dev));
      }
    }

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecB_array(const int b, const Gdims& _adims, const int maxl, const FILLTYPE fill, const int _dev=0){
      for(int l=0; l<=maxl; l++)
	parts.push_back(new SO3partB_array(b,_adims,l,2*l+1,fill,_dev));
    }

    
    // ---- Named constructors --------------------------------------------------------------------------------

    
    static SO3vecB_array zero(const Gdims& _adims, const SO3type& tau, const int _dev=0){
      return SO3vecB_array(_adims,tau,cnine::fill_zero(),_dev);
    }
  
    static SO3vecB_array gaussian(const Gdims& _adims, const SO3type& tau, const int _dev=0){
      return SO3vecB_array(_adims,tau,cnine::fill_gaussian(),_dev);
    }


    static SO3vecB_array Fzero(const Gdims& _adims, const int maxl, const int _dev=0){
      return SO3vecB_array(_adims,maxl,cnine::fill_zero(),_dev);
    }
  
    static SO3vecB_array Fgaussian(const Gdims& _adims, const int maxl, const int _dev=0){
      return SO3vecB_array(_adims,maxl,cnine::fill_gaussian(),_dev);
    }
  

    static SO3vecB_array zero(const int b, const Gdims& _adims, const SO3type& tau, const int _dev=0){
      return SO3vecB_array(b,_adims,tau,cnine::fill_zero(),_dev);
    }
  
    static SO3vecB_array gaussian(const int b, const Gdims& _adims, const SO3type& tau, const int _dev=0){
      return SO3vecB_array(b,_adims,tau,cnine::fill_gaussian(),_dev);
    }


    static SO3vecB_array Fzero(const int b, const Gdims& _adims, const int maxl, const int _dev=0){
      return SO3vecB_array(b,_adims,maxl,cnine::fill_zero(),_dev);
    }
  
    static SO3vecB_array Fgaussian(const int b, const Gdims& _adims, const int maxl, const int _dev=0){
      return SO3vecB_array(b,_adims,maxl,cnine::fill_gaussian(),_dev);
    }
  

    static SO3vecB_array zeros_like(const SO3vecB_array& x){
      return SO3vecB_array::zero(x.getb(),x.get_adims(),x.get_tau(),x.get_dev());
    }

    static SO3vecB_array gaussian_like(const SO3vecB_array& x){
      return SO3vecB_array::gaussian(x.getb(),x.get_adims(),x.get_tau(),x.get_dev());
    }


  public: // ---- Copying -------------------------------------------------------------------------------------------


    SO3vecB_array(const SO3vecB_array& x):
      SO3vecB_base(x){
      GELIB_COPY_WARNING();
    }

    SO3vecB_array(SO3vecB_array&& x):
      SO3vecB_base(std::move(x)){
      GELIB_MOVE_WARNING();
    }

    SO3vecB_array& operator=(const SO3vecB_array& x){
      SO3vecB_base::operator=(x);
      return *this;
    }

    SO3vecB_array& operator=(SO3vecB_array&& x){
      SO3vecB_base::operator=(std::move(x));
      return *this;
    }

    /*
    SO3vecB_array(const SO3vecB_array& x){
      GELIB_COPY_WARNING();
      for(auto& p:x.parts)
	parts.push_back(new SO3partB_array(*p));
    }

    SO3vecB_array(SO3vecB_array&& x){
      GELIB_MOVE_WARNING();
      parts=x.parts;
      x.parts.clear();

    }

    SO3vecB_array& operator=(const SO3vecB_array& x){
      GELIB_ASSIGN_WARNING();
      SO3vecB_base::operator=(x);
      return *this;
    }

    SO3vecB_array& operator=(SO3vecB_array&& x){
      GELIB_MASSIGN_WARNING();
      SO3vecB_base::operator=(std::move(x));
      return *this;
    }
    */


  public: // ---- Conversions ------------------------------------------------------------------------------------


    SO3vecB_array(const SO3vecB_base& x):
      SO3vecB_base(x){
      GELIB_CONVERT_WARNING(x);
    }

    SO3vecB_array(SO3vecB_base&& x):
      SO3vecB_base(std::move(x)){
      GELIB_MCONVERT_WARNING(x);
    }


  public: // ---- Views -------------------------------------------------------------------------------------------


    SO3vecB_array view(){
      return SO3vecB_array(SO3vecB_base::view());
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

    vector<at::Tensor> torch(){
      vector<at::Tensor> R;
      for(auto p: parts)
	R.push_back(p->torch());
      return R;
    }

#endif

  
  public: // ---- Access --------------------------------------------------------------------------------------------
  

    int getb() const{
      if(parts.size()>0) return parts[0]->getb();
      return 0;
    }

    Gdims get_adims() const{
      if(parts.size()>0) return parts[0]->get_adims();
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
      SO3vecB_array R=SO3vecB_array::zero(getb(),get_adims(),GElib::CGproduct(get_tau(),y.get_tau(),maxl),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }


    void add_CGproduct(const SO3vecB_array& x, const SO3vecB_array& y){
      assert(get_tau()==GElib::CGproduct(x.get_tau(),y.get_tau(),get_maxl()));
      LoggedTimer timer("CGproduct("+x.get_tau().str()+","+y.get_tau().str()+","+get_tau().str()+")[b="+to_string(getb())+","+get_adims().str()+",dev="+to_string(get_dev())+"]");

      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
      vector<int> offs(parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	//if(x.tau[l1]==0) continue;
	for(int l2=0; l2<=L2; l2++){
	  //if(y.tau[l2]==0) continue;
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l]->add_CGproduct(*x.parts[l1],*y.parts[l2],offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*(y.parts[l2]->getn());
	  }
	}
      }
    }

      
    void add_CGproduct_back0(const SO3vecB_array& g, const SO3vecB_array& y){
      assert(g.get_tau()==GElib::CGproduct(get_tau(),y.get_tau(),g.get_maxl()));
      LoggedTimer timer("CGproduct_back0("+get_tau().str()+","+y.get_tau().str()+","+g.get_tau().str()+")[b="+to_string(getb())+","+get_adims().str()+",dev="+to_string(get_dev())+"]");

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
      LoggedTimer timer("CGproduct_back1("+x.get_tau().str()+","+get_tau().str()+","+g.get_tau().str()+")[b="+to_string(getb())+","+get_adims().str()+",dev="+to_string(get_dev())+"]");

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

      
    // ---- Blocked CG-products ------------------------------------------------------------------------------


    SO3vecB_array BlockedCGproduct(const SO3vecB_array& y, const int bsize, const int maxl=-1) const{
      assert(get_adims()==y.get_adims());

      SO3vecB_array R=SO3vecB_array::zero(getb(),get_adims(),GElib::BlockedCGproduct(get_tau(),y.get_tau(),bsize,maxl),get_dev());
      R.add_BlockedCGproduct(*this,y,bsize);
      return R;
    }


    void add_BlockedCGproduct(const SO3vecB_array& x, const SO3vecB_array& y, const int bsize){
      assert(get_tau()==GElib::BlockedCGproduct(x.get_tau(),y.get_tau(),bsize,get_maxl()));

      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
      vector<int> offs(parts.size(),0);
	
      int count=0;
      SO3type tau=x.get_tau();
      for(int l1=0; l1<=L1; l1++)
	for(int l2=0; l2<=L2; l2++)
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++)
	    for(int i=-l1; i<=l1; i++) 
	      count+=tau[l1]*bsize*(std::min(l2,l-i)-std::max(-l2,-l-i)+(i<=l));
      count*=getb()*get_adims().total();

      LoggedTimer timer("  DiagCGproduct("+x.get_tau().str()+","+y.get_tau().str()+","+get_tau().str()+")[b="+
	to_string(x.getb())+",asize="+to_string(get_adims().total())+",blocksize="+to_string(bsize)+",maxl="+to_string(L)+
	",total="+to_string(count)+",dev="+to_string(get_dev())+"]",count);

      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l]->add_BlockedCGproduct(*x.parts[l1],*y.parts[l2],bsize,offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*bsize;
	  }
	}
      }
    }

      
    void add_BlockedCGproduct_back0(const SO3vecB_array& g, const SO3vecB_array& y, const int bsize){
      assert(g.get_tau()==GElib::BlockedCGproduct(get_tau(),y.get_tau(),bsize,g.get_maxl()));

      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
      vector<int> offs(g.parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l1]->add_BlockedCGproduct_back0(*g.parts[l],*y.parts[l2],bsize,offs[l]);
	    offs[l]+=(parts[l1]->getn())*bsize;
	  }
	}
      }
    }

      
    void add_BlockedCGproduct_back1(const SO3vecB_array& g, const SO3vecB_array& x, const int bsize){
      assert(g.get_tau()==GElib::BlockedCGproduct(x.get_tau(),get_tau(),bsize,g.get_maxl()));

      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=g.get_maxl();
      vector<int> offs(g.parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l2]->add_BlockedCGproduct_back1(*g.parts[l],*x.parts[l1],bsize,offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*bsize;
	  }
	}
      }
    }

      
    // ---- Diagonal CG-products -----------------------------------------------------------------------------


    SO3vecB_array DiagCGproduct(const SO3vecB_array& y, const int maxl=-1) const{
       return BlockedCGproduct(y,1,maxl);
    }

    void add_DiagCGproduct(const SO3vecB_array& x, const SO3vecB_array& y){
       add_BlockedCGproduct(x,y,1);
    }

    void add_DiagCGproduct_back0(const SO3vecB_array& g, const SO3vecB_array& y){
       add_BlockedCGproduct_back0(g,y,1);
    }

    void add_DiagCGproduct_back1(const SO3vecB_array& g, const SO3vecB_array& x){
      add_BlockedCGproduct_back1(g,x,1);
    }

    

    /*
    SO3vecB_array DiagCGsquare(const int maxl=-1) const{
      SO3vecB_array R=SO3vecB_array::zero(getb(),GElib::DiagCGproduct(get_tau(),get_tau(),1,get_maxl()),get_dev());
      R.add_DiagCGsquare(*this);
      return R;
    }

    void add_DiagCGsquare(const SO3vecB_array& x){
      GELIB_ASSRT(get_tau()==GElib::DiagCGproduct(x.get_tau(),x.get_tau(),get_maxl()));

      int L1=x.get_maxl(); 
      int L=get_maxl();
      vector<int> offs(parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L1; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l]->add_DiagCGproduct(*x.parts[l1],*x.parts[l2],offs[l]);
	    offs[l]+=(x.parts[l1]->getn());
	  }
	}
      }
    }
      
    void add_DiagCGsquare_back(const SO3vecB_array& g, const SO3vecB_array& y, const int bsize){
      GELIB_ASSRT(y.get_tau()==get_tau());
      GELIB_ASSRT(g.get_tau()==GElib::DiagCGproduct(get_tau(),get_tau(),g.get_maxl()));

      int L1=get_maxl(); 
      int L=g.get_maxl();
      vector<int> offs(g.parts.size(),0);
      vector<int> offs2(g.parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L1; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l1]->add_DiagCGproduct_back0(*g.parts[l],*y.parts[l2],offs[l]);
	    parts[l2]->add_DiagCGproduct_back1(*g.parts[l],*y.parts[l1],offs2[l]);
	    offs[l]+=(parts[l1]->getn())*bsize;
	    offs2[l]+=(parts[l2]->getn())*bsize;
	  }
	}
      }
    }
    */


    // ---- Diagonal CG-squares ------------------------------------------------------------------------------

    // ---- DDiag CG-products ------------------------------------------------------------------------------


    SO3vecB_array DDiagCGproduct(const SO3vecB_array& y, const int maxl=-1) const{
      SO3vecB_array R=SO3vecB_array::zero(getb(),get_adims(),GElib::DDiagCGproduct(get_tau(),maxl),get_dev());
      R.add_DDiagCGproduct(*this,y);
      return R;
    }

    void add_DDiagCGproduct(const SO3vecB_array& x, const SO3vecB_array& y){
      assert(x.get_adims()==y.get_adims());
      assert(x.get_tau()==y.get_tau());
      assert(get_tau()==GElib::DDiagCGproduct(x.get_tau(),get_maxl()));

      int L1=x.get_maxl(); 
      int L=get_maxl();
      vector<int> offs(parts.size(),0);
	
      for(int l=0; l<=L1 && l+l%2<=L; l++){
	parts[l+l%2]->add_BlockedCGproduct(*x.parts[l],*y.parts[l],1,offs[l+l%2]);
	offs[l+l%2]+=x.parts[l]->getn();
      }
    }

    void add_DDiagCGproduct_back0(const SO3vecB_array& g, const SO3vecB_array& y){
      assert(get_adims()==y.get_adims());
      assert(get_tau()==y.get_tau());
      assert(g.get_tau()==GElib::DDiagCGproduct(get_tau(),g.get_maxl()));

      int L1=get_maxl(); 
      vector<int> offs(g.parts.size(),0);
	
      for(int l=0; l<=L1; l++){
	parts[l]->add_BlockedCGproduct_back0(*g.parts[l+l%2],*y.parts[l],1,offs[l+l%2]);
	offs[l+l%2]+=(parts[l]->getn());
      }
    }

    void add_DDiagCGproduct_back1(const SO3vecB_array& g, const SO3vecB_array& x){
      assert(get_adims()==x.get_adims());
      assert(get_tau()==x.get_tau());
      assert(g.get_tau()==GElib::DDiagCGproduct(get_tau(),g.get_maxl()));

      int L1=get_maxl(); 
      vector<int> offs(g.parts.size(),0);
	
      for(int l=0; l<=L1; l++){
	parts[l]->add_BlockedCGproduct_back1(*g.parts[l+l%2],*x.parts[l],1,offs[l+l%2]);
	offs[l+l%2]+=(parts[l]->getn());
      }
    }
      

  public: // ---- CG-squares ----------------------------------------------------------------------------------


    SO3vecB_array CGsquare(const int maxl=-1) const{
      SO3vecB_array R=SO3vecB_array::zero(getb(),get_adims(),GElib::CGsquare(get_tau(),maxl),get_dev());
      R.add_CGsquare(*this);
      return R;
    }


    void add_CGsquare(const SO3vecB_array& x){
      assert(get_tau()==GElib::CGsquare(x.get_tau(),get_maxl()));

      int L1=x.get_maxl(); 
      int L=get_maxl();
      vector<int> offs(parts.size(),0);
	
      for(int l1=0; l1<=L1; l1++){

	for(int l=0; l<=2*l1 && l<=L; l++){
	  int n=x.parts[l1]->getn();
	  parts[l]->add_CGsquare(*x.parts[l1],offs[l]);
	  offs[l]+=(n*(n-1))/2+n*(1-l%2);
	}

	for(int l2=l1+1; l2<=L1; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    parts[l]->add_CGproduct(*x.parts[l1],*x.parts[l2],offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*(x.parts[l2]->getn());
	  }
	}
	
      }
    }


  public: // ---- Fproducts ----------------------------------------------------------------------------------


    SO3vecB_array Fproduct(const SO3vecB_array& y, int maxl=-1) const{
      assert(y.get_adims()==get_adims());
      if(maxl<0) maxl=get_maxl()+y.get_maxl();
      SO3vecB_array R=SO3vecB_array::Fzero(getb(),get_adims(),maxl,get_dev());
      R.add_Fproduct(*this,y);
      return R;
    }


    SO3vecB_array FproductB(const SO3vecB_array& y, int maxl=-1) const{
      assert(y.get_adims()==get_adims());
      if(maxl<0) maxl=get_maxl()+y.get_maxl();
      SO3vecB_array R=SO3vecB_array::Fzero(getb(),get_adims(),maxl,get_dev());
      R.add_FproductB(*this,y);
      return R;
    }


    void add_Fproduct(const SO3vecB_array& x, const SO3vecB_array& y, const int method=0){
      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L ; l++){
	    SO3part_addFproduct_Fn(0,method)(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3());
	  }
	}
      }
    }

    
    void add_FproductB(const SO3vecB_array& x, const SO3vecB_array& y){
      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
	
      if(get_dev()==0){
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L ; l++){
	    SO3part_addFproduct_Fn()(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3());
	  }
	}
      }
      }

      #ifdef _WITH_CUDA
      if(get_dev()==1){
	for(int l=0; l<=L1+L2 && l<=L ; l++){
	  cudaStream_t stream;
	  CUDA_SAFE(cudaStreamCreate(&stream));
	  for(int l1=std::max(0,l-L2); l1<=std::min(l+L2,L1); l1++){
	    for(int l2=std::abs(l-l1); l2<=std::min(l+l1,L2); l2++){
	      SO3Fpart_addFproduct_cu(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3(),0,0,stream);
	    }
	  }
	  CUDA_SAFE(cudaStreamSynchronize(stream));
	  CUDA_SAFE(cudaStreamDestroy(stream));
	}
      }
      #endif 
    }


    void add_Fproduct_back0(const SO3vecB_array& g, const SO3vecB_array& y, const int method=0){
      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    SO3part_addFproduct_back0Fn(0,method)(parts[l1]->view3(),g.parts[l]->view3(),y.parts[l2]->view3());
	  }
	}
      }
    }


    void add_Fproduct_back1(const SO3vecB_array& g, const SO3vecB_array& x){
      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    SO3part_addFproduct_back1Fn()(parts[l2]->view3(),g.parts[l]->view3(),x.parts[l1]->view3());
	  }
	}
      }
    }

    
  public: // ---- Fmodsq -------------------------------------------------------------------------------------
    

    SO3vecB_array Fmodsq(int maxl=-1) const{
      if(maxl<0) maxl=2*get_maxl();
      SO3vecB_array R=SO3vecB_array::Fzero(getb(),get_adims(),maxl,get_dev());
      R.add_Fmodsq(*this,*this);
      return R;
    }


    void add_Fmodsq(const SO3vecB_array& x, const SO3vecB_array& y){
      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L ; l++){
	    SO3part_addFproduct_Fn(1)(parts[l]->view3(),x.parts[l1]->view3(),y.parts[l2]->view3().flip());
	  }
	}
      }
    }

    void add_Fmodsq_back(const SO3vecB_array& g, const SO3vecB_array& x){
      add_Fmodsq_back0(g,x);
      add_Fmodsq_back1(g,x);
    }

    void add_Fmodsq_back0(const SO3vecB_array& g, const SO3vecB_array& y){
      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    SO3part_addFproduct_back0Fn(1)(parts[l1]->view3(),g.parts[l]->view3(),y.parts[l2]->view3().flip());
	  }
	}
      }
    }

    void add_Fmodsq_back1(const SO3vecB_array& g, const SO3vecB_array& x){
      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    SO3part_addFproduct_back1Fn(1)(parts[l2]->view3().flip(),g.parts[l]->view3(),x.parts[l1]->view3());
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

    static string classname(){
      return "GElib::SO3vecB_array";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3vecB_array of type("+to_string(getb())+","+get_adims().str()+","+get_tau().str()+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3vecB_array& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif

