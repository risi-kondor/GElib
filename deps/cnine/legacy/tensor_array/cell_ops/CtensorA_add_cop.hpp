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


#ifndef _CtensorA_add_cop
#define _CtensorA_add_cop

#include "GenericCop.hpp"


namespace cnine{

#ifdef _WITH_CUDA
  template<typename CMAP>
  void CtensorA_add_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream);
#endif 


  class CtensorA_add_cop: public UnaryCop<CtensorA,CtensorArrayA>{
  public:

    CtensorA_add_cop(){}
    
    void operator()(CtensorA& r, const CtensorA& y) const{
      r.add(y);
    }

    template<typename IMAP>
    void operator()(const IMAP& map, CtensorArrayA& r, const CtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      CtensorA_add_cu(map,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      CNINE_NOCUDA_ERROR;
#endif
    }

  };


  class CtensorA_subtract_cop: public UnaryCop<CtensorA,CtensorArrayA>{
  public:

    CtensorA_subtract_cop(){}
    
    void operator()(CtensorA& r, const CtensorA& y) const{
      r.subtract(y);
    }

    template<typename IMAP>
    void operator()(const IMAP& map, CtensorArrayA& r, const CtensorArrayA& x) const{
      CNINE_UNIMPL();
    }


  };

}

#endif
