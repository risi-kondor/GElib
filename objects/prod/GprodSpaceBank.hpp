// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GprodSpaceBank
#define _GprodSpaceBank

#include "GprodSpaceObj.hpp"


namespace GElib{

  template<typename GROUP>
  class GprodSpaceProductSignature: public pair<int,int>{
  public:
    GprodSpaceProductSignature(const GprodSpaceObj<GROUP>& x, const GprodSpaceObj<GROUP>& y):
      pair<int,int>(x.id,y.id){}
  };


  template<typename GROUP>
  class GprodSpaceBank{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef GprodSpaceProductSignature<GROUP> _psignature;


    int nspaces=0;
    map<_IrrepIx,GprodSpaceObj<GROUP>*> singletons;
    map<_psignature,GprodSpaceObj<GROUP>*> products;



  public: // ---- Access -------------------------------------------------------------------------------------


    GprodSpaceObj<GROUP>* operator()(const _IrrepIx& ix){
      auto it=singletons.find(ix);
      if(it!=singletons.end()) return it->second;
      GprodSpaceObj<GROUP>* x=new GprodSpaceObj<GROUP>(ix,nspaces++);
      singletons[ix]=x;
      return x;
    }

    GprodSpaceObj<GROUP>* operator()(GprodSpaceObj<GROUP>* left, GprodSpaceObj<GROUP>* right){
      _psignature ix(*left,*right);
      auto it=products.find(ix);
      if(it!=products.end()) return it->second;
      GprodSpaceObj<GROUP>* x=new GprodSpaceObj<GROUP>(left,right,nspaces++);
      products[ix]=x;
      return x;
    }

  };


}

#endif 
