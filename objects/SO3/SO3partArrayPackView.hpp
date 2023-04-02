
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partArrayPackView
#define _GElibSO3partArrayPackView

#include "GElib_base.hpp"
#include "TensorPackView.hpp"
#include "TensorArrayPackView.hpp"
#include "SO3partArrayView.hpp"


namespace GElib{

  template<typename RTYPE>
  class SO3partArrayPackView: public cnine::TensorArrayPackView<complex<RTYPE> >{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;
    typedef cnine::Gdims Gdims;

    //typedef cnine::TensorPackView<complex<RTYPE> > TensorPackView;
    typedef cnine::TensorArrayPackView<complex<RTYPE> > TensorArrayPackView;

    using TensorArrayPackView::TensorArrayPackView;
    using TensorArrayPackView::dims;
    using TensorArrayPackView::strides;
    using TensorArrayPackView::arr;
    using TensorArrayPackView::size;


  public: // ---- Constructors --------------------------------------------------------------------------------

  public: // ---- Access --------------------------------------------------------------------------------------


    SO3partArrayView<RTYPE> operator[](const int i) const{
      return SO3partArrayView<RTYPE>(arr,dims(i),strides(i));
    }


    //SO3partArrayView<RTYPE> operator()(const int i){
    //return cnine::TensorView<complex<RTYPE> >(arr,dims(i),strides(i));
    //}


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "SO3partArrayPackView";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Array "<<i<<":"<<endl;
	oss<<(*this)[i].str(indent+"  ")<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const SO3partArrayPackView& v){
      stream<<v.str(); return stream;}


    

  };

}

#endif
