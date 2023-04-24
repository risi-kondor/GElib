
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
#include "BatchedTensorView.hpp"
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
  class SO3partView: public cnine::BatchedTensorView<complex<RTYPE> >{
  public:

    typedef cnine::BatchedTensorView<complex<RTYPE> > TensorView;

    using TensorView::TensorView;
    using TensorView::arr;
    using TensorView::dims;
    using TensorView::strides;

    using TensorView::device;
    using TensorView::bbatch;
    using TensorView::getb;
    

  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partView(const TensorView& x):
      TensorView(x){}

    //SO3partView(const cnine::TensorView<complex<RTYPE> >& x):
    //TensorView(x){}

    operator SO3part3_view() const{
      return SO3part3_view(arr.template ptr_as<RTYPE>(),{dims[0],dims[1],dims[2]},
	{2*strides[0],2*strides[1],2*strides[2]},1,device());
    }


  public: // ---- Access --------------------------------------------------------------------------------------

    
    int getl() const{
      return (dims[1]-1)/2;
    }

    int getn() const{
      return dims[2];
    }

    SO3partView<RTYPE> batch(const int i) const{
      return bbatch(i);
      //return SO3partView<RTYPE>(arr+strides[0]*i,dims.chunk(1),strides.chunk(1));
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    void add_CGproduct(const SO3partView& x, const SO3partView& y, const int _offs=0) const{
      cnine::reconcile_batches<SO3partView>(*this,x,y,
	[&](const auto& r, const auto& x, const auto& y){SO3part_addCGproductFn()(r,x,y,_offs);},
	[&](const auto& r, const auto& x, const auto& y){SO3part_addRCGproductFn()(r,x,y,_offs);});
    }

    void add_CGproduct_back0(const SO3partView& g, const SO3partView& y, const int _offs=0){
      cnine::reconcile_batches<SO3partView>(*this,g,y,
	[&](const auto& xg, const auto& g, const auto& y){SO3part_addCGproduct_back0Fn()(xg,g,y,_offs);},
	[&](const auto& xg, const auto& g, const auto& y){SO3part_addRCGproduct_back0Fn()(xg,g,y,_offs);});
    }

    void add_CGproduct_back1(const SO3partView& g, const SO3partView& x, const int _offs=0){
      cnine::reconcile_batches<SO3partView>(*this,g,x,
	[&](const auto& yg, const auto& g, const auto& x){SO3part_addCGproduct_back1Fn()(yg,g,x,_offs);},
	[&](const auto& yg, const auto& g, const auto& x){SO3part_addRCGproduct_back1Fn()(yg,g,x,_offs);});
    }



    void add_DiagCGproduct(const SO3partView& x, const SO3partView& y, const int _offs=0) const{
      cnine::reconcile_batches<SO3partView>(*this,x,y,
	[&](const auto& r, const auto& x, const auto& y){SO3part_addBlockedCGproductFn()(r,x,y,1,_offs);},
	[&](const auto& r, const auto& x, const auto& y){
	  GELIB_UNIMPL();
	  //SO3part_addRCGproductFn()(r,x,y,_offs);
	});
    }

    void add_DiagCGproduct_back0(const SO3partView& g, const SO3partView& y, const int _offs=0){
      cnine::reconcile_batches<SO3partView>(*this,g,y,
	[&](const auto& xg, const auto& g, const auto& y){SO3part_addBlockedCGproduct_back0Fn()(xg,g,y,1,_offs);},
	[&](const auto& xg, const auto& g, const auto& y){
	  GELIB_UNIMPL();
	  //SO3part_addRCGproduct_back0Fn()(xg,g,y,_offs);
	});
    }

    void add_DiagCGproduct_back1(const SO3partView& g, const SO3partView& x, const int _offs=0){
      cnine::reconcile_batches<SO3partView>(*this,g,x,
	[&](const auto& yg, const auto& g, const auto& x){SO3part_addBlockedCGproduct_back1Fn()(yg,g,x,1,_offs);},
	[&](const auto& yg, const auto& g, const auto& x){
	  GELIB_UNIMPL();
	  //SO3part_addRCGproduct_back1Fn()(yg,g,x,_offs);
	});
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string repr(const string indent="") const{
      return "<GElib::SO3part(b="+to_string(getb())+",l="+to_string(getl())+",n="+to_string(getn())+")>";
    }

    //friend ostream& operator<<(ostream& stream, const SO3partView& x){
    //stream<<x.str(); return stream;
    //}
    
  };

}


#endif 




   /*
    string str(const string indent="") const{
      ostringstream oss;
      if(getb()>1){
	for_each_batch([&](const int b, const VEC& x){
	    oss<<indent<<"Batch "<<b<<":"<<endl;
	    oss<<indent+"  "<<x<<endl;
	  });
	return;
      }
      for_each_part([&](const int p, const Pview& x){
	  oss<<indent<<"Part "<<p<<":"<<endl;
	  oss<<indent<<x<<endl;
	});
      return oss.str();
    }
    */

