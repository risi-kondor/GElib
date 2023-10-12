
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partView
#define _GElibSO3partView

#include "GElib_base.hpp"
#include "LtensorViewSub.hpp"
#include "TensorTemplates.hpp"
#include "SO3part3_view.hpp"
#include "SO3templates.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"

#include "SO3part_addRCGproductFn.hpp"
#include "SO3part_addRCGproduct_back0Fn.hpp"
#include "SO3part_addRCGproduct_back1Fn.hpp"

#include "SO3part_addBlockedCGproductFn.hpp"
#include "SO3part_addBlockedCGproduct_back0Fn.hpp"
#include "SO3part_addBlockedCGproduct_back1Fn.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3partView: public cnine::LtensorView<complex<RTYPE> >{
  public:

    typedef cnine::LtensorView<complex<RTYPE>,SO3partView> BASE;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;

    using BASE::device;
    using BASE::bbatch;
    using BASE::getb;
    

  public: // ---- Conversions --------------------------------------------------------------------------------


    //SO3partView(const BASE& x):
    //BASE(x){}

    //SO3partView(const cnine::BASE<complex<RTYPE> >& x):
    //BASE(x){}

    operator cnine::Ctensor3_view() const{
      return cnine::Ctensor3_view(arr.template ptr_as<RTYPE>(),{dims[0],dims[1],dims[2]},
	cnine::GstridesB(2*strides[0],2*strides[1],2*strides[2]),1,device());
    }

    operator SO3part3_view() const{
      return SO3part3_view(arr.template ptr_as<RTYPE>(),{dims[0],dims[1],dims[2]},
	cnine::GstridesB(2*strides[0],2*strides[1],2*strides[2]),1,device());
    }


  public: // ---- Access --------------------------------------------------------------------------------------

    
    int getl() const{
      return (dims[1]-1)/2;
    }

    int getn() const{
      return dims[2];
    }



  public: // ---- I/O ----------------------------------------------------------------------------------------


    string repr(const string indent="") const{
      return "<GElib::SO3part(b="+to_string(getb())+",l="+to_string(getl())+",n="+to_string(getn())+")>";
    }

    /*
    string str(const string indent="") const{
      if(dev>0){
	auto t=cnine::BatchedTensor<complex<RTYPE> >(*this,0);
	return SO3partView(t).str(indent);
      }
      return cnine::BatchedTensorView<complex<RTYPE> >::str(indent);
    }
    */

    //friend ostream& operator<<(ostream& stream, const SO3partView& x){
    //stream<<x.str(); return stream;
    //}
    
  };

}


#endif 

