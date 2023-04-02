
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partArrayView
#define _GElibSO3partArrayView

#include "GElib_base.hpp"
#include "TensorView.hpp"
#include "SO3partView.hpp"

namespace GElib{

  template<typename RTYPE>
  class SO3partArrayView: public cnine::TensorView<complex<RTYPE> >{
  public:

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;
    typedef cnine::TensorView<complex<RTYPE> > TensorView;

    using TensorView::dims;
    using TensorView::strides;
    using TensorView::dev;

    using TensorView::TensorView;
    using TensorView::ndims;
    using TensorView::slice;

    
  public: // ---- Access --------------------------------------------------------------------------------------
    

  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "SO3partArrayView";
    }

    string describe() const{
      ostringstream oss;
      oss<<"SO3partArrayView"<<dims<<" "<<strides<<""<<endl;
      return oss.str();
    }

    string str(const string indent="") const{
      CNINE_CPUONLY();
      GELIB_ASSRT(ndims()>2);
      ostringstream oss;

      Gdims adims=dims.chunk(0,ndims()-2);
      adims.for_each_index([&](const Gindex& ix){
	  oss<<indent<<"Cell"<<ix<<":"<<endl;
	  oss<<slice(ix).str(indent+"  ")<<endl;
	});

      return oss.str();
    }

  };

}


#endif 
