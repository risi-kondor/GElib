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


#ifndef _CtensorA_copy_cop
#define _CtensorA_copy_cop

#include "GenericCop.hpp"
#include "Cmaps2.hpp"


namespace cnine{

#ifdef _WITH_CUDA

  template<typename CMAP>
  void CtensorA_copy_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream);

  template<typename CMAP>
  void CtensorA_add_copy_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream);

  template<typename CMAP>
  void CtensorA_copy_accumulator_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream);

#endif 


  class CtensorA_copy_cop{
  public:

    CtensorA_copy_cop(){}


    void apply(CtensorA& r, const CtensorA& x) const{
      r=x;
    }

    void add(CtensorA& r, const CtensorA& x) const{
      r.add(x);
    }


    template<typename CMAP>
    void apply(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
	CtensorA_copy_cu(cmap,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }

    template<typename CMAP>
    void add(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
	CtensorA_add_copy_cu(cmap,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }

    template<typename CMAP>
    void accumulate(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
	CtensorA_copy_accumulator_cu(cmap,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }


  };


}

#endif


  //template<typename CMAP, typename = typename std::enable_if<std::is_base_of<DirectCmap, CMAP>::value, CMAP>::type>
  //void CtensorA_copy_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream);

  //template<typename CMAP, typename = typename std::enable_if<std::is_base_of<AccumulatorCmap, CMAP>::value, CMAP>::type>
  //void CtensorA_copy_cu(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x, const cudaStream_t& stream);

    /*
    template<typename CMAP, typename = typename std::enable_if<std::is_base_of<AccumulatorCmap, CMAP>::value, CMAP>::type>
    void operator()(const CMAP& map, CtensorArrayA& r, const CtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
      CtensorA_copy_accumulator_cu(map,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#else
      CNINE_NOCUDA_ERROR;
#endif
    }
    */
    /*
    template<typename CMAP>
    void operator()(const CMAP& cmap, CtensorArrayA& r, const CtensorArrayA& x) const{
#ifdef _WITH_CUDA
      cudaStream_t stream;
      CUDA_SAFE(cudaStreamCreate(&stream));
	CtensorA_copy_cu(cmap,r,x,stream);
      CUDA_SAFE(cudaStreamSynchronize(stream));
      CUDA_SAFE(cudaStreamDestroy(stream));
#endif
    }
    */

    /*
    void operator()(CtensorA& r, const CtensorA& x) const{
      r=x;
    }
    */

