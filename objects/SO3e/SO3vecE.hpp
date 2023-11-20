// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3vecE
#define _SO3vecE

#include "SO3groupE.hpp"
#include "GtypeE.hpp"
#include "GvecSpec.hpp"
#include "GvecE.hpp"


namespace GElib{


  template<typename TYPE>
  class SO3vecE;


  class SO3typeE: public GtypeE{
  public:

    typedef GtypeE BASE;

    //SO3typeE():
    //BASE(new SO3group()){}

    SO3typeE(const initializer_list<int>& list){
      int l=0;
      for(auto p:list)
	(*this)[SO3irrepIx(l++)]=p;
    }

  };


  // ---------------------------------------------------------------------------------------------------------


  template<typename TYPE>
  class SO3vecSpec: public GvecSpecBase<SO3vecSpec<TYPE> >{
  public:

    typedef GvecSpecBase<SO3vecSpec<TYPE> > BASE;

    using BASE::ddims;
    using BASE::ix;

    SO3vecSpec():
      BASE(new SO3group()){}

    SO3vecSpec(const BASE& x): 
      BASE(x){
      if(ddims.size()!=2) ddims=cnine::Gdims(0,0);
    }

    SO3vecE<TYPE> operator ()() const{
      return SO3vecE<TYPE>(*this);
    }

    SO3vecSpec& l(const int _l){
      ix.reset(new SO3irrepIx(_l));
      ddims[0]=2*_l+1;
      return *this;
    }

  };


  // ---------------------------------------------------------------------------------------------------------


  template<typename TYPE>
  class SO3vecE: public GvecE<complex<TYPE> >{
  public:

    typedef GvecE<complex<TYPE> > BASE;
    using BASE::BASE;

    SO3vecE(const BASE& x):
      BASE(x){}

    SO3vecE(const SO3vecSpec<TYPE>& x):
      BASE(x){}
    

  public: // ---- SO3vecSpec -------------------------------------------------------------------------------

    static SO3vecSpec<TYPE> raw() {return SO3vecSpec<TYPE>().raw();}
    static SO3vecSpec<TYPE> zero() {return SO3vecSpec<TYPE>().zero();}
    static SO3vecSpec<TYPE> sequential() {return SO3vecSpec<TYPE>().sequential();}
    static SO3vecSpec<TYPE> gaussian() {return SO3vecSpec<TYPE>().gaussian();}


  };

}

#endif 
