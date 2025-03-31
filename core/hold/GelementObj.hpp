// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibGelementObj
#define _GElibGelementObj

#include "GElib_base.hpp"


namespace GElib{

  //class GgroupObj;


  class GelementObj{
  public:

    //shared_ptr<GgroupObj> G;

    virtual ~GelementObj(){}


  public: // ---- Constructors -------------------------------------------------------------------------------


    //GelementObj(const shared_ptr<GgroupObj>& _G):
    //G(_G){}


  public: // ---- Operations ---------------------------------------------------------------------------------


    virtual bool operator==(const GelementObj& y) const=0;

    virtual shared_ptr<GelementObj> identity() const=0;

    virtual shared_ptr<GelementObj> mult(const GelementObj& y) const=0;

    virtual shared_ptr<GelementObj> pow(const int m) const{
      auto R=identity();
      if(m>=0){
	for(int i=0; i<m; i++)
	  R=R->mult(*this);
      }else{
	auto xinv=inv();
	for(int i=0; i<m; i++)
	  R=R->mult(*xinv);
      }
      return R;
    }

    virtual shared_ptr<GelementObj> inv() const=0;

    
  public: // ---- I/O -----------------------------------------------------------------------------------------


    virtual string repr() const=0;

    virtual string str(const string indent="") const=0;

  };

}

#endif 


    //virtual GelementObj* clone() const=0;
