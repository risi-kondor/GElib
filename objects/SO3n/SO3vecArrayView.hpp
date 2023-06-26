// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3vecArrayView
#define _GElibSO3vecArrayView

#include "GElib_base.hpp"
#include "GvecArrayView.hpp"
#include "SO3type.hpp"
#include "SO3partArrayView.hpp"
#include "SO3vecView.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3vecArrayView: public GvecArrayView<int,SO3partArrayView<RTYPE>,SO3vecArrayView<RTYPE>,SO3vecView<RTYPE> >{
  public:

    typedef GvecArrayView<int,SO3partArrayView<RTYPE>,SO3vecArrayView<RTYPE>,SO3vecView<RTYPE> > _GvecArrayView;
    typedef SO3partArrayView<RTYPE> _SO3partArrayView;

    using _GvecArrayView::_GvecArrayView;
    using _GvecArrayView::parts;
    using _GvecArrayView::getb;
    using _GvecArrayView::get_adims;
    using _GvecArrayView::device;

#ifdef _WITH_ATEN
    using _GvecArrayView::torch;
#endif


  public: // ---- Access ------------------------------------------------------------------------------------


    int get_maxl() const{
      int r=0;
      for(auto& p:parts)
	r=std::max(r,p.first);
      return r;
    }

    SO3type get_tau() const{
      SO3type tau(parts.size(),cnine::fill_raw());
      for(auto& p:parts)
	tau[p.first]=p.second->getn();
      return tau;
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


    void add_CGproduct(const SO3vecArrayView& x, const SO3vecArrayView& y){
      vCGproduct<SO3vecArrayView,_SO3partArrayView>(*this,x,y,
	[&](const _SO3partArrayView& r, const _SO3partArrayView& x, const _SO3partArrayView& y, const int offs){
	  r.add_CGproduct(x,y,offs);});
    }

    void add_CGproduct_back0(const SO3vecArrayView& g, const SO3vecArrayView& y){
      vCGproduct<SO3vecArrayView,_SO3partArrayView>(g,*this,y,
	[&](const _SO3partArrayView& g, const _SO3partArrayView& gx, const _SO3partArrayView& y, const int offs){
	  gx.add_CGproduct(g,y,offs);});
    }

    void add_CGproduct_back1(const SO3vecArrayView& g, const SO3vecArrayView& x){
      vCGproduct<SO3vecArrayView,_SO3partArrayView>(g,x,*this,
	[&](const _SO3partArrayView& g, const _SO3partArrayView& x, const _SO3partArrayView& gy, const int offs){
	  gy.add_CGproduct(g,x,offs);});
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    static string classname(){
      return "GElib::SO3vecArrayView";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3vecArr b="+to_string(getb())+", adims="+get_adims().str()+", tau="+get_tau().str()+">";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3vecArrayView& x){
      stream<<x.str(); return stream;
    }

    

  };


}

#endif 


    /*
    string str(const string indent="") const{
      ostringstream oss;
	for(int l=0; l<parts.size(); l++){
	  //if(!parts[l]) continue;
	  oss<<indent<<"Part l="<<l<<":\n";
	  oss<<(*this)(l).str(indent+"  ");
	  oss<<endl;
	}
      return oss.str();
    }
    */
