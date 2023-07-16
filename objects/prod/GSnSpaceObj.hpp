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


namespace GElib{

  template<typename GROUP>
  class GSnSpaceObj{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;

    int id=0;
    _IrrepIx irrep;
    GSnSpaceObj* left=nullptr;
    GSnSpaceObj* right=nullptr;


    GSnSpaceObj(_IrrepIx _irrep, const int _id): 
      id(_id), irrep(_irrep){}

    GSnSpaceObj(GSnSpaceObj* x, GSnSpaceObj* y, const int _id): 
      id(_id), left(x), right(y){}


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
      return "";
    }

    friend ostream& operator<<(ostream& stream, const GSnSpaceObj& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
