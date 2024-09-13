// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2024, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _TensorUtils
#define _TensorUtils

#include "BatchedTensor.hpp"

namespace GElib{

  template<typename TYPE>
  int tile_channels(cnine::TensorView<TYPE>& x, const int n){
    GELIB_ASSRT(x.ndims()==4);
    int remainder=x.dims[3]-((x.dims[3])/n)*n;
    x.dims=Gdims({x.dims[0],x.dims[1],x.dims[2],(x.dims[3])/n,n});
    return remainder;
  }


  // Bring a tensor to GElib's 1+2 or 1+1+2 standard format
  template<typename TYPE>
  cnine::BatchedTensor<TYPE> canonicalize(const cnine::TensorView<TYPE>& x){
    int d=x.ndims();
    if(d==3) return x;
    if(d>3) return x.fuse_chunk(1,d-3);

    cnine::TensorView<TYPE> r(x);
    if(d==1){
      r.dims=r.dims.insert(0,1);
      r.strides=r.strides.insert(0,0);
    }
    if(d==2){
      r.dims=r.dims.insert(2,1);
      r.strides=r.strides.insert(2,0);
    }
    return r;
  }

  // unused
  // Bring a tensor to GElib's 1+1+2 format
  template<typename TYPE>
  cnine::TensorView<TYPE> canonicalize_to_4d(const cnine::TensorView<TYPE>& x){
    int d=x.ndims();
    if(d==4) return x;
    if(d>4) return x.fuse_chunk(1,d-3);

    cnine::TensorView<TYPE> r(x);
    if(d==1){
      r.dims=cnine::Gdims({1,1,x.dims[0],1});
      r.strides=cnine::GstridesB({0,0,x.strides[0],0});
    }
    if(d==2){
      r.dims=cnine::Gdims({1,1,x.dims[0],x.dims[1]});
      r.strides=cnine::GstridesB({0,0,x.strides[0],x.strides[1]});
    }
    if(d==3){
      r.dims=cnine::Gdims({x.dims[0],1,x.dims[1],x.dims[2]});
      r.strides=cnine::GstridesB({x.strides[0],0,x.strides[1],x.strides[2]});
    }
    return r;
  }

  
  /*
  template<typename GPART>
  GPART fuse_grid(){
    if(!is_grid()){
      GPART r(*this);
      r.dims=r.dims.insert(1,1);
      r.strides=r.strides.insert(1,0);
      return r;
      }
    return static_cast<const GPART&>(*this).like(fuse_chunk(1,ndims()-3));
  }
  */

  // This replaces the analogous method in Gpart
  template<typename TYPE1, typename TYPE2>
  void for_each_cell_multi(const cnine::BatchedTensor<TYPE1>& r, const cnine::BatchedTensor<TYPE2>& x, 
    const std::function<void(const int, const int, const cnine::TensorView<TYPE1>& r, const cnine::TensorView<TYPE2>& x)>& lambda){

    if(r.ndims()==3 && x.ndims()==3){
      r.template for_each_batch_multi<TYPE2>(x,[&](const int b, const cnine::TensorView<TYPE1>& r, const cnine::TensorView<TYPE2>& x){
	  lambda(b,0,r,x);
	});
      return;
    }

    auto _r=fuse_grid(r);
    auto _x=canonicalize(x);
    int G=std::max(_r.dims[1],_x.dims[1]);
    GELIB_ASSRT(_r.dims[0]==G || _r.dims[0]==1);
    GELIB_ASSRT(_x.dims[0]==G || _x.dims[0]==1);
    int mr=(_r.dims[0]>1);
    int mx=(_x.dims[0]>1);
    _r.template for_each_batch_multi<TYPE2>(_x,[&](const int b, const cnine::TensorView<TYPE1>& r, 
	const cnine::TensorView<TYPE2>& x){
	for(int g=0; g<G; g++)
	  lambda(b,g,r.slice(0,mr*g),x.slice(0,mx*g));
      });
  }


}


#endif 
