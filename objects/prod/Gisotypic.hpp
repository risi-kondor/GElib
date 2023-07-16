// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _Gisotypic
#define _Gisotypic


namespace GElib{

  template<typename GROUP>
  class Gisotypic{
  public:

    typedef typename GROUP::IrrepIx _IrrepIx;

    _IrrepIx ix;
    int m=1;
    
    Gisotypic(const _IrrepIx& _ix, const int _m=1):
      ix(_ix), m(_m){}


  public: // ---- I/O ---------------------------------------------------------------------------------------

    string repr() const{
      ostringstream oss;
      oss<<"Isotypic<"<<GROUP::repr()<<">("<<ix<<","<<m<<")";
      return oss.str();
    }

    string str(const string indent="") const{
      return indent+repr();
    }

    friend ostream& operator<<(ostream& stream, const Gisotypic& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
