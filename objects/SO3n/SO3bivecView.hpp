// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3bivecView
#define _GElibSO3bivecView

#include "GElib_base.hpp"
#include "GvecView.hpp"
#include "SO3bipartView.hpp"
#include "SO3bitype.hpp"
#include "SO3vecView.hpp"
#include "SO3templates.hpp"


namespace GElib{


  template<typename RTYPE>
  class SO3bivecView: public GvecView<pair<int,int>,SO3bipartView<RTYPE>,SO3bivecView<RTYPE> >{
  public:

    typedef GvecView<pair<int,int>,SO3bipartView<RTYPE>,SO3bivecView<RTYPE> > _GvecView;
    typedef SO3partView<RTYPE> _SO3partView;

    using _GvecView::_GvecView;
    using _GvecView::parts;
    using _GvecView::size;
    using _GvecView::getb;
    using _GvecView::str;


  public: // ---- Access ------------------------------------------------------------------------------------


    //int get_maxl() const{
    //int r=0;
    //for(auto& p:parts)
    //r=std::max(r,p.first);
    //return r;
    //}
    
    SO3bitype get_tau() const{
      SO3bitype tau;
      for(auto& p:parts)
	tau[p.first]=p.second->getn();
      return tau;
    }

    SO3bipartView<RTYPE> part(const int l1, const int l2) const{
      return *parts[pair<int,int>(l1,l2)];
    }

    // this is a hack for the decltype in Gvec
    SO3bipartView<RTYPE> part(const int dummy) const{
      return SO3bipartView<RTYPE>();
    }


  public: // ---- CG-transform ------------------------------------------------------------------------------


    void add_CGtransform_to(const SO3vecView<RTYPE>& r) const{
      int L=r.get_maxl();
      vector<int> offs(L+1,0);
 
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


    void add_CGtransform_back(const SO3vecView<RTYPE>& r) const{
      int L=r.get_maxl();
      vector<int> offs(L+1,0);
 
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
      return "GElib::SO3bivecView";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3bivec b="+to_string(getb())+", tau="+get_tau().str()+">";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3bivecView& x){
      stream<<x.str(); return stream;
    }

    

  };


}

#endif 


    /*
    string str(const string indent="") const{
      ostringstream oss;
      for_each_batch([&](const int b, const VEC& x){
	  oss<<indent<<"Batch "<<b<<":"<<endl;
	  oss<<indent<<x<<endl;
	});
      //for(int l=0; l<parts.size(); l++){
	//if(!parts[l]) continue;
      //oss<<indent<<"Part l="<<l<<":\n";
      //oss<<(*this)(l).str(indent+"  ");
      //oss<<endl;
      //}
      return oss.str();
    }
    */

