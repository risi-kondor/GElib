
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Fvec
#define _SO3Fvec

#include "GElib_base.hpp"
#include "SO3Fpart.hpp"
#include "SO3vecB.hpp"


namespace GElib{


  class SO3Fvec: public SO3vecB{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorPackObj ctensorpack;



    //vector<SO3Fpart*> parts;

    SO3Fvec(){}

    //~SO3vec(){
    //for(auto p: parts) delete p;  
    //}


    // ---- Constructors --------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3Fvec(const int b, const int maxl, const FILLTYPE fill, const int _dev){
      for(int l=0; l<=maxl; l++)
	parts.push_back(new SO3Fpart(b,l,fill,_dev));
    }
    
    
    // ---- Named constructors --------------------------------------------------------------------------------

    
    static SO3Fvec zero(const int b, const int maxl, const int _dev=0){
      return SO3Fvec(b,maxl,cnine::fill_zero(),_dev);
    }

    static SO3Fvec gaussian(const int b, const int maxl, const int _dev=0){
      return SO3Fvec(b,maxl,cnine::fill_gaussian(),_dev);
    }

    
    // ---- Copying -------------------------------------------------------------------------------------------



    SO3Fvec(const SO3Fvec& x):
      SO3vecB(x){}

    SO3Fvec(SO3Fvec&& x):
      SO3vecB(std::move(x)){}

    /*
    SO3Fvec(const SO3Fvec& x){
      for(auto& p:x.parts)
	parts.push_back(p)
    }

    SO3Fvec(SO3Fvec&& x){
      parts=x.parts;
      x.parts.clear();
    }
    */


    // ---- Access --------------------------------------------------------------------------------------------


    //int getb() const{
    //if(parts.size()>0) return parts[0].getn();
    //return 0;
    //}

    //SO3type get_tau() const{
    //SO3type tau;
    //for(auto& p:parts)
    //tau.push_back(p.getn());
    //return tau;
    //}

    //int get_maxl() const{
    //return parts.size();
    //}

    //int get_dev() const{
    //if(parts.size()>0) return parts[0].get_dev();
    //return 0;
    //}



    // ---- CG-products ---------------------------------------------------------------------------------------


    SO3Fvec Fproduct(const SO3Fvec& y, int maxl=-1){
      assert(y.getb()==getb());

      if(maxl<0) maxl=get_maxl()+y.get_maxl();
      SO3Fvec R=SO3Fvec::zero(getb(),maxl,get_dev());
      R.add_Fproduct(*this,y);
      return R;
    }


    void add_Fproduct(const SO3Fvec& x, const SO3Fvec& y){
      assert(x.getb()==getb());
      assert(y.getb()==getb());
      
      int L1=x.get_maxl(); 
      int L2=y.get_maxl();
      int L=get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L ; l++){
	    as_SO3Fpart(*parts[l]).add_Fproduct(as_SO3Fpart(*x.parts[l1]),as_SO3Fpart(*y.parts[l2]));
	  }
	}
      }
    }


    void add_Fproduct_back0(const SO3Fvec& g, const SO3Fvec& y){
      assert(g.getb()==getb());
      assert(y.getb()==getb());
      
      int L1=get_maxl(); 
      int L2=y.get_maxl();
      int L=g.get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    as_SO3Fpart(*parts[l1]).add_Fproduct_back0(as_SO3Fpart(*g.parts[l]),as_SO3Fpart(*y.parts[l2]));
	  }
	}
      }
    }


    void add_Fproduct_back1(const SO3Fvec& g, const SO3Fvec& x){
      assert(g.getb()==getb());
      assert(x.getb()==getb());
      
      int L1=x.get_maxl(); 
      int L2=get_maxl();
      int L=get_maxl();
	
      for(int l1=0; l1<=L1; l1++){
	for(int l2=0; l2<=L2; l2++){
	  for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	    as_SO3Fpart(*parts[l2]).add_Fproduct(as_SO3Fpart(*g.parts[l]),as_SO3Fpart(*x.parts[l1]));
	  }
	}
      }
    }

  };

}

#endif
