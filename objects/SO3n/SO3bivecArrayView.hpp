// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3bivecArrayView
#define _GElibSO3bivecArrayView

#include "GElib_base.hpp"
#include "GvecArrayView.hpp"
#include "SO3bitype.hpp"
#include "SO3bipartArrayView.hpp"
#include "SO3bivecView.hpp"
#include "SO3vecArrayView.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3bivecArrayView: 
    public GvecArrayView<pair<int,int>, SO3bipartArrayView<RTYPE>, SO3bivecArrayView<RTYPE>, SO3bivecView<RTYPE> >{
  public:

    typedef GvecArrayView<pair<int,int>,SO3bipartArrayView<RTYPE>,SO3bivecArrayView<RTYPE>,SO3bivecView<RTYPE> > _GvecArrayView;
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


    SO3bitype get_tau() const{
      SO3bitype tau;
      for(auto& p:parts)
	tau[p.first]=p.second->getn();
      return tau;
    }

    // this is a hack for the decltype in Gvec
    SO3bipartArrayView<RTYPE> part(const int dummy) const{
      return SO3bipartArrayView<RTYPE>();
    }


  public: // ---- CG-products --------------------------------------------------------------------------------


  public: // ---- CG-transform ------------------------------------------------------------------------------


    void add_CGtransform_to(const SO3vecArrayView<RTYPE>& r) const{
      int L=r.get_maxl();
      vector<int> offs(L+1,0);
 
      int count=0;
      for(auto& p:parts){
	auto& P=*p.second;
	int l1=P.getl1();
	int l2=P.getl2();
	for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++)
	  for(int m1=-l1; m1<=l1; m1++) 
	    count+=P.getn()*(std::min(l2,l-m1)-std::max(-l2,-l-m1)+(m1<=l));
      }
      count*=getb()*get_adims().total();

      LoggedTimer timer("  CGtransform("+get_tau().str()+","+r.get_tau().str()+")[b="+
	to_string(getb())+",adims="+get_adims().str()+",maxl="+to_string(L)+
	",total="+to_string(count)+",dev="+to_string(device())+"]",count);

      for(auto& p:parts){
	auto& P=*p.second;
	int l1=P.getl1();
	int l2=P.getl2();
	for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	  P.add_CGtransform_to(r.part(l),offs[l]);
	  offs[l]+=P.getn();
	}
      }
    }


    void add_CGtransform_back(const SO3vecArrayView<RTYPE>& r) const{
      int L=r.get_maxl();
      vector<int> offs(L+1,0);
 
      int count=0;
      for(auto& p:parts){
	auto& P=*p.second;
	int l1=P.getl1();
	int l2=P.getl2();
	for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++)
	  for(int m1=-l1; m1<=l1; m1++) 
	    count+=P.getn()*(std::min(l2,l-m1)-std::max(-l2,-l-m1)+(m1<=l));
      }
      count*=getb()*get_adims().total();

      LoggedTimer timer("  CGtransform_back("+get_tau().str()+","+r.get_tau().str()+")[b="+
	to_string(getb())+",adims="+get_adims().str()+",maxl="+to_string(L)+
	",total="+to_string(count)+",dev="+to_string(device())+"]",count);

      for(auto& p:parts){
	auto& P=*p.second;
	int l1=P.getl1();
	int l2=P.getl2();
	for(int l=std::abs(l2-l1); l<=l1+l2 && l<=L; l++){
	  P.add_CGtransform_back(r.part(l),offs[l]);
	  offs[l]+=P.getn();
	}
      }
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    static string classname(){
      return "GElib::SO3bivecArrayView";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3vecArr b="+to_string(getb())+", adims="+get_adims().str()+", tau="+get_tau().str()+">";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3bivecArrayView& x){
      stream<<x.str(); return stream;
    }

    

  };


}

#endif 

