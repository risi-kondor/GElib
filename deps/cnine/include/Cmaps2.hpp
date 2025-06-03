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


#ifndef _Cnine_Cmaps2
#define _Cnine_Cmaps2

#include "Cnine_base.hpp"

#ifdef _WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>
#endif 

namespace cnine{


  class Cmap_base{
  public:

#ifdef _WITH_CUDA

    dim3 blockdims() const{
      return dim3(0);
    }

    __device__ int n_accum(const int b) const{
      return 0;
    }

    __device__ int target(const int b) const{
      return 0;
    }

    __device__ int lst_ptr(const int b) const{
      return 0;
    }

    __device__ thrust::tuple<int,int,int> operator()(const int i, const int j, const int k) const{
      return thrust::make_tuple(0,0,0);
    }

    __device__ thrust::tuple<int,int> source(const int lst, const int b, const int i) const{
      return thrust::make_tuple(0,0);
    }

#endif 

  };


  class Direct_cmap: public Cmap_base{
  public:

  };

  class Masked2_cmap: public Cmap_base{
  public:

  };



  /*
  class DirectCmap{
  public:
  };

  class UnaryDirectCmap: public DirectCmap{
  public:
  };

  class BinaryDirectCmap: public DirectCmap{
  public:
  };



  class AccumulatorCmap{
  public:
  };
  */
  




}

#endif 
