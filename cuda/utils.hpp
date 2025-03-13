/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GElib_cuda_utils_hpp
#define _GElib_cuda_utils_hpp

#include <cuda.h>
#include <cuda_runtime.h>
#include "Ctensor3_view.hpp"
#include "Ctensor4_view.hpp"
#include "Ctensor5_view.hpp"
#include "Ctensor6_view.hpp"

#define tix threadIdx.x

namespace GElib{


  inline int roundup32(const int x){
    return ((x-1)/32+1)*32;
  }


  inline std::pair<int,int> optimal_tile_size(int nx, int ny){
    if(nx*ny<=1024)
      return make_pair(nx,ny);
    if(ny<=1024)
      return make_pair(1024/ny,ny);
    return make_pair(1,1024);
  }


  inline cnine::Ctensor4_view view4_of(const cnine::TensorView<complex<float> >& x){
    GELIB_ASSRT(x.ndims()==4);
    return cnine::Ctensor4_view(x.mem_as<float>(),x.mem_as<float>()+1,
      x.dim(0),x.dim(1),x.dim(2),x.dim(3),
      2*x.stride(0),2*x.stride(1),2*x.stride(2),2*x.stride(3),x.get_dev());
  }

  inline cnine::Ctensor5_view view5_of(const cnine::TensorView<complex<float> >& x){
    GELIB_ASSRT(x.ndims()==5);
    return cnine::Ctensor5_view(x.mem_as<float>(),x.mem_as<float>()+1,
      x.dim(0),x.dim(1),x.dim(2),x.dim(3),x.dim(4),
      2*x.stride(0),2*x.stride(1),2*x.stride(2),2*x.stride(3),2*x.stride(4),x.get_dev());
  }

  inline cnine::Ctensor5_view tiled_view4_of(const cnine::TensorView<complex<float> >& x, const int n){
    GELIB_ASSRT(x.ndims()==4);
    return cnine::Ctensor5_view(x.mem_as<float>(),x.mem_as<float>()+1,
      x.dim(0),x.dim(1),x.dim(2),x.dim(3)/n,n,
      2*x.stride(0),2*x.stride(1),2*x.stride(2),2*x.stride(3)*n,2*x.stride(3),x.get_dev());
  }

  inline cnine::Ctensor6_view tiled_view5_of(const cnine::TensorView<complex<float> >& x, const int n){
    GELIB_ASSRT(x.ndims()==5);
    return cnine::Ctensor6_view(x.mem_as<float>(),x.mem_as<float>()+1,
      x.dim(0),x.dim(1),x.dim(2),x.dim(3),x.dim(4)/n,n,
      2*x.stride(0),2*x.stride(1),2*x.stride(2),2*x.stride(3),2*x.stride(4)*n,2*x.stride(4),x.get_dev());
  }

}


#endif 


  /*
  void optimal_tile_size(int& tx, int& ty, int nx, int ny){
    int ux=roundup32(nx);
    int uy=roundup32(ny);

    if(ux*uy<=1024){
      tx=nx;
      ty=ny;
      return;
    }

    if(ny<=1024){
      tx=std::min(1024/uy,nx);
      ty=ny;
      return;
    }

    tx=1;
    ty=std::min(1024,ny)
  }


  inline Ctensor4_view tile3(const int& last, const Ctensor3_view& x, const int n){
    Ctensor4_view R(x.arr,x.arrc,x.n0,x.n1,(x.n2-1)/n+1,n,x.s0,x.s1,x.s2*n,x.s2,x.dev);
    last=x.n2-((x.n2-1)/n)*n;
    return R;
  }
  */
