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


#ifndef _CnineCUDA_helpers
#define _CnineCUDA_helpers

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cnine_base.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"


namespace cnine{ 


  inline void dispatch(const Rtensor2_view& r, const Rtensor2_view& x, 
    std::function<void(const dim3& blocks, const dim3& threads, const int, const int, const int, const int)> tt_fn,
    std::function<void(const dim3& blocks, const dim3& threads, const int, const int, const int, const int)> bt_fn,
    std::function<void(const dim3& blocks, const dim3& threads, const int, const int, const int, const int)> bb_fn){

    const int n0=x.n0;
    const int n1=x.n1;

    if(n0*n1<=1024){
      dim3 blocks(1);
      dim3 threads(n0,n1);
      tt_fn(blocks,threads,x.s0,x.s1,r.s0,r.s1);
      return;
    }

    if(n1<=1024){
      dim3 blocks(n0);
      dim3 threads(n1);
      bt_fn(blocks,threads,x.s0,x.s1,r.s0,r.s1);
      return;
    }

    if(n0<=1024){
      dim3 blocks(n1);
      dim3 threads(n0);
      bt_fn(blocks,threads,x.s1,x.s0,r.s1,r.s0);
      return;
    }

    dim3 blocks(n0,n1);
    dim3 threads(1);
    bb_fn(blocks,threads,x.s0,x.s1,r.s0,r.s1);
    return;
  }


  inline void dispatch(const Rtensor3_view& r, const Rtensor3_view& x, 
    std::function<void(const dim3& blocks, const dim3& threads, const int, const int, const int, const int, const int, const int)> ttt_fn,
    std::function<void(const dim3& blocks, const dim3& threads, const int, const int, const int, const int, const int, const int)> btt_fn,
    std::function<void(const dim3& blocks, const dim3& threads, const int, const int, const int, const int, const int, const int)> bbt_fn,
    std::function<void(const dim3& blocks, const dim3& threads, const int, const int, const int, const int, const int, const int)> bbb_fn){

    const int n0=x.n0;
    const int n1=x.n1;
    const int n2=x.n2;

    if(n0*n1*n2<=1024){
      dim3 blocks(1);
      dim3 threads(n0,n1,n2);
      ttt_fn(blocks,threads,x.s0,x.s1,x.s2,r.s0,r.s1,r.s2);
      return;
    }

    if(n1*n2<=1024){
      dim3 blocks(n0);
      dim3 threads(n1,n2);
      btt_fn(blocks,threads,x.s0,x.s1,x.s2,r.s0,r.s1,r.s2);
      return;
    }

    if(n0*n2<=1024){
      dim3 blocks(n1);
      dim3 threads(n0,n2);
      btt_fn(blocks,threads,x.s1,x.s0,x.s2,r.s1,r.s0,r.s2);
      return;
    }

    if(n0*n1<=1024){
      dim3 blocks(n2);
      dim3 threads(n0,n1);
      btt_fn(blocks,threads,x.s2,x.s0,x.s1,r.s2,r.s0,r.s1);
      return;
    }

    if(n2<=1024){
      dim3 blocks(n0,n1);
      dim3 threads(n2);
      bbt_fn(blocks,threads,x.s0,x.s1,x.s2,r.s0,r.s0,r.s2);
      return;
    }

    if(n1<=1024){
      dim3 blocks(n0,n2);
      dim3 threads(n1);
      bbt_fn(blocks,threads,x.s1,x.s0,x.s2,r.s1,r.s0,r.s2);
      return;
    }

    if(n0<=1024){
      dim3 blocks(n1,n2);
      dim3 threads(n0);
      bbt_fn(blocks,threads,x.s2,x.s0,x.s1,r.s2,r.s0,r.s1);
      return;
    }

    dim3 blocks(n0,n1,n2);
    dim3 threads(1);
    bbb_fn(blocks,threads,x.s0,x.s1,x.s1,r.s0,r.s1,r.s1);
    return;
  }

}


#endif 
