/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineSparseIndexerBase
#define _CnineSparseIndexerBase

#include "Cnine_base.hpp"
#include "TensorView.hpp"

namespace cnine{


  class SparseIndexerBase{
  public:

    virtual int dsparse() const=0;

    virtual int nfilled() const=0;

    virtual int offset(const int i0) const{
      CNINE_UNIMPL();
      return 0;
    }

    virtual int offset(const int i0, const int i1) const{
      CNINE_UNIMPL();
      return 0;
    }

    virtual int offset(const int i0, const int i1, const int i2) const{
      CNINE_UNIMPL();
      return 0;
    }

    virtual int offset(const Gindex& x) const=0;

    virtual void for_each(std::function<void(const Gindex&, const int)> lambda) const {CNINE_UNIMPL();};
    //virtual void for_each(std::function<void(const int, const int)> lambda) const {CNINE_UNIMPL();};
    virtual void for_each(std::function<void(const int, const int, const int)> lambda) {CNINE_UNIMPL();};
    virtual void for_each(std::function<void(const int, const int, const int, const int)> lambda) const {CNINE_UNIMPL();};

  };

}

#endif 

