
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3bipartView
#define _GElibSO3bipartView

#include "GElib_base.hpp"
#include "BatchedTensorView.hpp"
#include "TensorTemplates.hpp"
#include "Ctensor4_view.hpp"

#include "SO3partView.hpp"
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

#include "SO3part_addCGtransformFn.hpp"
#include "SO3part_addCGtransform_backFn.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3bipartView: public cnine::BatchedTensorView<complex<RTYPE> >{
  public:

    typedef cnine::BatchedTensorView<complex<RTYPE> > Tview;

    using Tview::Tview;
    using Tview::arr;
    using Tview::dims;
    using Tview::strides;

    using Tview::device;
    using Tview::bbatch;
    using Tview::getb;
    using Tview::for_each_batch;
    

  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3bipartView(const Tview& x):
      Tview(x){}

    //SO3bipartView(const cnine::Tview<complex<RTYPE> >& x):
    //Tview(x){}

    operator cnine::Ctensor4_view() const{
      return cnine::Ctensor4_view(arr.template ptr_as<RTYPE>(),{dims[0],dims[1],dims[2],dims[3]},
	{2*strides[0],2*strides[1],2*strides[2],2*strides[3]},1,device());
    }


  public: // ---- Access --------------------------------------------------------------------------------------

    
    int getl1() const{
      return (dims[1]-1)/2;
    }

    int getl2() const{
      return (dims[2]-1)/2;
    }

    int getn() const{
      return dims[3];
    }

    SO3bipartView<RTYPE> batch(const int i) const{
      return bbatch(i);
      //return SO3bipartView<RTYPE>(arr+strides[0]*i,dims.chunk(1),strides.chunk(1));
    }


  public: // ---- CG-transforms ------------------------------------------------------------------------------


    void add_CGtransform_to(const SO3partView<RTYPE>& r, const int offs=0) const{
      SO3part_addCGtransformFn()(cnine::Ctensor3_view(r),cnine::Ctensor4_view(*this),offs);
    }

    void add_CGtransform_back(const SO3partView<RTYPE>& r, const int offs=0) const{
      SO3part_addCGtransform_backFn()(cnine::Ctensor4_view(*this),cnine::Ctensor3_view(r),offs);
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------


    string repr(const string indent="") const{
      return "<GElib::SO3bipart(b="+to_string(getb())+",l1="+to_string(getl1())+",l2="+to_string(getl2())+
	",n="+to_string(getn())+")>";
    }

    string str(const string indent="") const{
      if(device()>0){
	auto t=cnine::BatchedTensor<complex<RTYPE> >(*this,0);
	return SO3bipartView(t).str(indent);
      }
      if(getb()>1){
	ostringstream oss;
	for_each_batch([&](const int b, const SO3bipartView& x){
	    oss<<indent<<"Batch "<<b<<":"<<endl;
	    oss<<x.str(indent+"  ")<<endl; 
	  });
	return oss.str();
      }
      ostringstream oss;
      auto x=Tview::batch(0);
      for(int i=0; i<getn(); i++){
	oss<<indent<<"  "<<"Channel "<<i<<":"<<endl;
	oss<<x.slice(2,i).str(indent+"  ")<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SO3bipartView& x){
      stream<<x.str(); return stream;
    }
    
  };

}


#endif 


