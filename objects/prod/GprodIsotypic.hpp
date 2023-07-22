// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GprodIsotypic
#define _GprodIsotypic

#include "Lmatrix.hpp"


namespace GElib{

  template<typename GROUP>
  class GprodIsotypic{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef cnine::Lmatrix<_IrrepIx,_IrrepIx,int> _Lmatrix;


    _IrrepIx ix;
    int n=0;
    _Lmatrix* offsets=nullptr;;
    
    ~GprodIsotypic(){
      delete offsets;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    GprodIsotypic(){}

    GprodIsotypic(const _IrrepIx& _ix, const int _n=0):
      ix(_ix), n(_n){}

    //GprodIsotypic(const _IrrepIx& _ix, const cnine::Llist<_IrrepIx>& _llabels, const cnine::Llist<_IrrepIx>& _rlabels):
    //ix(_ix), offsets(new _Lmatrix(_llabels,_rlabels,cnine::fill_constant<int>(-1))){}


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string repr() const{
      ostringstream oss;
      oss<<"Isotypic<"<<GROUP::repr()<<">("<<ix<<","<<n<<")";
      return oss.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent+repr()<<endl;
      //if(offsets) oss<<offsets->str(indent+"  ");
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GprodIsotypic& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
