// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _GSnSpaceObj
#define _GSnSpaceObj

#include "Gisotypic.hpp"


namespace GElib{

  template<typename GROUP>
  class GSnSpaceObj{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;
    typedef Gisotypic<GROUP> _Isotypic;

    int id=0;
    _IrrepIx irrep;
    GSnSpaceObj* left=nullptr;
    GSnSpaceObj* right=nullptr;
    map<_IrrepIx,_Isotypic*> isotypics;

    ~GSnSpaceObj(){
      for(auto p:isotypics) delete p.second;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    GSnSpaceObj(_IrrepIx _irrep, const int _id): 
      id(_id), irrep(_irrep){
      isotypics[_irrep]=new _Isotypic(_irrep);
    }

    GSnSpaceObj(GSnSpaceObj* _x, GSnSpaceObj* _y, const int _id): 
      id(_id), left(_x), right(_y){
      for(auto x:_x->isotypics)
	for(auto y:_y->isotypics)
	  GROUP::for_each_CGcomponent(x.second->ix,y.second->ix,[&](const _IrrepIx& _irrep, const int m){
	  auto it=isotypics.find(_irrep);
	  if(it!=isotypics.end()) it->second->m+=m*x.second->m*y.second->m;
	  else isotypics[_irrep]=new _Isotypic(_irrep,m);
	});
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    GSnSpaceObj(const GSnSpaceObj& x)=delete;


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string reprr() const{
      ostringstream oss;
      if(!left) oss<<"("<<irrep<<")";
      else oss<<"("<<left->reprr()<<"*"<<right->reprr()<<")";
      return oss.str();
    }

    string repr() const{
      ostringstream oss;
      oss<<"Gspace<"<<GROUP::repr()<<">"<<reprr();
      //if(!left) oss<<"Gspace<"<<GROUP::repr()<<">("<<irrep<<")";
      //else oss<<"("<<left->repr()<<"*"<<right->repr()<<")";
      return oss.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<repr()<<endl;
      for(auto p:isotypics)
	oss<<indent<<"  "<<*p.second<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GSnSpaceObj& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
