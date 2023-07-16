// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GSnSpaceBank
#define _GSnSpaceBank

#include "GSnSpaceObj.hpp"


namespace GElib{

  template<typename GROUP>
  class GSnSpaceProductSignature: public pair<int,int>{
  public:
    GSnSpaceProductSignature(const GSnSpaceObj<GROUP>& x, const GSnSpaceObj<GROUP>& y):
      pair<int,int>(x.id,y.id){}
  };


  template<typename GROUP>
  class GSnSpaceBank{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    //typedef GSnSpaceObj<GROUP> _GSnSpaceObj;
    typedef GSnSpaceProductSignature<GROUP> _psignature;
    //typedef typename GROUP::_SnSpaceObj _SnSpaceObj;

    int nspaces=0;
    map<_IrrepIx,GSnSpaceObj<GROUP>*> singletons;
    map<_psignature,GSnSpaceObj<GROUP>*> products;



  public: // ---- Access -------------------------------------------------------------------------------------


    GSnSpaceObj<GROUP>* operator()(const _IrrepIx& ix){
      auto it=singletons.find(ix);
      if(it!=singletons.end()) return it->second;
      GSnSpaceObj<GROUP>* x=new GSnSpaceObj<GROUP>(ix,nspaces++);
      singletons[ix]=x;
      return x;
    }

    GSnSpaceObj<GROUP>* operator()(GSnSpaceObj<GROUP>* left, GSnSpaceObj<GROUP>* right){
      _psignature ix(*left,*right);
      auto it=products.find(ix);
      if(it!=products.end()) return it->second;
      GSnSpaceObj<GROUP>* x=new GSnSpaceObj<GROUP>(left,right,nspaces++);
      products[ix]=x;
      return x;
    }

  };


}

#endif 
