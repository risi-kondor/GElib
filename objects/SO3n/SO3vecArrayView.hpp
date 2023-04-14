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
  class SO3vecArrayView: public GvecArrayView<int,SO3partArrayView<RTYPE>,SO3vecView<RTYPE> >{
  public:

    typedef GvecArrayView<int,SO3partArrayView<RTYPE>,SO3vecView<RTYPE> > GvecArrayView;
    typedef SO3partArrayView<RTYPE> SO3partArrayView;

    using GvecArrayView::GvecArrayView;
    using GvecArrayView::parts;


  public: // ---- Access ------------------------------------------------------------------------------------


  public: // ---- I/O ---------------------------------------------------------------------------------------


    static string classname(){
      return "GElib::SO3vecArrayView";
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
    
    friend ostream& operator<<(ostream& stream, const SO3vecArrayView& x){
      stream<<x.str(); return stream;
    }

    

  };


}

#endif 
    //SO3partArrayView operator()(const int l) const{
    //auto it=parts.find(l);
    //assert(it!=parts.end());
    //return SO3partArrayView(*it->second);
    //}


