// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3vecView
#define _GElibSO3vecView

#include "GElib_base.hpp"
#include "GvecView.hpp"
#include "SO3type.hpp"
#include "SO3partView.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3vecView: public GvecView<int,SO3partView<RTYPE> >{
  public:

    typedef GvecView<int,SO3partView<RTYPE> > GvecView;
    typedef SO3partView<RTYPE> SO3partView;

    using GvecView::GvecView;
    using GvecView::parts;


  public: // ---- Access ------------------------------------------------------------------------------------

    
    //SO3partView operator()(const int l) const{
    //auto it=parts.find(l);
    //assert(it!=parts.end());
    //return SO3partView(*it->second);
    //}

    //SO3partView part(const int l) const{
    //auto it=parts.find(l);
    //assert(it!=parts.end());
    //return SO3partView(*it->second);
    //}


  public: // ---- I/O ---------------------------------------------------------------------------------------


    static string classname(){
      return "GElib::SO3vecView";
    }

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

    string repr(const string indent="") const{
      return "";
      //return "<GElib::SO3vecV of type "+get_tau().str()+">";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3vecView& x){
      stream<<x.str(); return stream;
    }

  };


}

#endif 
