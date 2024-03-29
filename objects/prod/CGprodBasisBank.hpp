// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _CGprodBasisBank
#define _CGprodBasisBank

#include "CGprodBasisObj.hpp"


namespace GElib{

  template<typename GROUP>
  class GprodSpaceProductSignature: public pair<int,int>{
  public:
    GprodSpaceProductSignature(const CGprodBasisObj<GROUP>& x, const CGprodBasisObj<GROUP>& y):
      pair<int,int>(x.id,y.id){}
  };


  template<typename GROUP>
  class CGprodBasisBank{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef GprodSpaceProductSignature<GROUP> _psignature;


    int nspaces=0;
    map<_IrrepIx,CGprodBasisObj<GROUP>*> singletons;
    map<_psignature,CGprodBasisObj<GROUP>*> products;



  public: // ---- Access -------------------------------------------------------------------------------------


    CGprodBasisObj<GROUP>* operator()(const _IrrepIx& ix){
      auto it=singletons.find(ix);
      if(it!=singletons.end()) return it->second;
      CGprodBasisObj<GROUP>* x=new CGprodBasisObj<GROUP>(ix,nspaces++);
      singletons[ix]=x;
      return x;
    }

    CGprodBasisObj<GROUP>* operator()(CGprodBasisObj<GROUP>* left, CGprodBasisObj<GROUP>* right){
      _psignature ix(*left,*right);
      auto it=products.find(ix);
      if(it!=products.end()) return it->second;
      CGprodBasisObj<GROUP>* x=new CGprodBasisObj<GROUP>(left,right,nspaces++);
      products[ix]=x;
      return x;
    }

  };


}

#endif 
