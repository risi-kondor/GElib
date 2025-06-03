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

#ifndef _RtensorEinsumProducts_cu
#define _RtensorEinsumProducts_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Cnine_base.hpp"


template<int TIX>
__device__ void REP_getix(int& i0);

template<>
__device__ void REP_getix<0>(int& i0){
  i0=blockIdx.x;
}

template<>
__device__ void REP_getix<1>(int& i0){
  i0=threadIdx.x;
}


template<int TIX>
__device__ void REP_getix(int& i0, int& i1);

template<>
__device__ void REP_getix<0>(int& i0, int& i1){
  i0=blockIdx.x;
  i1=blockIdx.y;
}

template<>
__device__ void REP_getix<1>(int& i0, int& i1){
  i0=blockIdx.x;
  i1=threadIdx.x;
}

template<>
__device__ void REP_getix<2>(int& i0, int& i1){
  i0=threadIdx.x;
  i1=threadIdx.y;
}


template<int TIX>
__device__ void REP_getix(int& i0, int& i1, int& i2);

template<>
__device__ void REP_getix<0>(int& i0, int& i1, int& i2){
  i0=blockIdx.x;
  i1=blockIdx.y;
  i2=blockIdx.z;
}

template<>
__device__ void REP_getix<1>(int& i0, int& i1, int& i2){
  i0=blockIdx.x;
  i1=blockIdx.y;
  i2=threadIdx.x;
}

template<>
__device__ void REP_getix<2>(int& i0, int& i1, int& i2){
  i0=blockIdx.x;
  i1=threadIdx.x;
  i2=threadIdx.y;
}

template<>
__device__ void REP_getix<3>(int& i0, int& i1, int& i2){
  i0=threadIdx.x;
  i1=threadIdx.y;
  i2=threadIdx.z;
}


  // ---- 0 summation loops ----------------------------------------------------------------------------------


template<int TIX>
__global__ void Rtensor_addEinsum_0_0_0_kernel(const float* xarr, const float* yarr, float* rarr){
  rarr[0]+=xarr[0]*yarr[0];
}

template<int TIX>
__global__ void Rtensor_addEinsum_1_0_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int y0, const int r0){
  int i0;
  REP_getix<TIX>(i0);
  rarr[i0*r0]+=xarr[i0*x0]*yarr[i0*y0];
}

template<int TIX>
__global__ void Rtensor_addEinsum_2_0_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int x1, 
  const int y0, const int y1, 
  const int r0, const int r1){
  int i0,i1;
  REP_getix<TIX>(i0,i1);
  rarr[i0*r0+i1*r1]+=xarr[i0*x0+i1*x1]*yarr[i0*y0+i1*y1];
}

template<int TIX>
__global__ void Rtensor_addEinsum_3_0_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int x1, const int x2, 
  const int y0, const int y1, const int y2, 
  const int r0, const int r1, const int r2){
  int i0,i1,i2;
  REP_getix<TIX>(i0,i1,i2);
  rarr[i0*r0+i1*r1+i2*r2]+=xarr[i0*x0+i1*x1+i2*x2]*yarr[i0*y0+i1*y1+i2*y2];
}


// ---- 1 summation loop -----------------------------------------------------------------------------------


template<int TIX>
__global__ void Rtensor_addEinsum_0_1_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, 
  const int y0, 
  const int J0){
  float t=0;
  for(int j0=0; j0<J0; j0++)
    t+=xarr[j0*x0]*yarr[j0*y0];
  rarr[0]+=t;
}

template<int TIX>
__global__ void Rtensor_addEinsum_1_1_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int x1, 
  const int y0, const int y1, 
  const int r0, 
  const int J0){
  int i0;
  REP_getix<TIX>(i0);
  float t=0;
  for(int j0=0; j0<J0; j0++)
    t+=xarr[i0*x0+j0*x1]*yarr[i0*y0+j0*y1];
  rarr[i0*r0]+=t;
}

template<int TIX>
__global__ void Rtensor_addEinsum_2_1_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int x1, const int x2,  
  const int y0, const int y1, const int y2, 
  const int r0, const int r1,
  const int J0){
  int i0,i1;
  REP_getix<TIX>(i0,i1);
  float t=0;
  for(int j0=0; j0<J0; j0++)
    t+=xarr[i0*x0+i1*x1+j0*x2]*yarr[i0*y0+i1*y1+j0*y2];
  rarr[i0*r0+i1*r1]+=t;
}

template<int TIX>
__global__ void Rtensor_addEinsum_3_1_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int x1, const int x2, const int x3,
  const int y0, const int y1, const int y2, const int y3, 
  const int r0, const int r1, const int r2, 
  const int J0){
  int i0,i1,i2;
  REP_getix<TIX>(i0,i1,i2);

  float t=0;
  for(int j0=0; j0<J0; j0++)
    t+=xarr[i0*x0+i1*x1+i2*x2+j0*x3]*yarr[i0*y0+i1*y1+i2*y2+j0*y3];
  rarr[i0*r0+i1*r1+i2*r2]+=t; 
}


// ---- 2 summation loops ----------------------------------------------------------------------------------


template<int TIX>
__global__ void Rtensor_addEinsum_0_2_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int x1, 
  const int y0, const int y1, 
  const int J0, const int J1){
  float t=0;
  for(int j0=0; j0<J0; j0++)
    for(int j1=0; j1<J1; j1++)
      t+=xarr[j0*x0+j1*x1]*yarr[j0*y0+j1*x1];
  rarr[0]+=t;
}

template<int TIX>
__global__ void Rtensor_addEinsum_1_2_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int x1, const int x2, 
  const int y0, const int y1, const int y2, 
  const int r0, 
  const int J0, const int J1){
  int i0;
  REP_getix<TIX>(i0);
  float t=0;
  for(int j0=0; j0<J0; j0++)
    for(int j1=0; j1<J1; j1++)
      t+=xarr[i0*x0+j0*x1+j1*x2]*yarr[i0*y0+j0*y1+j1*y2];
  rarr[i0*r0]+=t;
}

template<int TIX>
__global__ void Rtensor_addEinsum_2_2_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int x1, const int x2, const int x3,  
  const int y0, const int y1, const int y2, const int y3, 
  const int r0, const int r1,
  const int J0, const int J1){
  int i0,i1;
  REP_getix<TIX>(i0,i1);
  float t=0;
  for(int j0=0; j0<J0; j0++)
    for(int j1=0; j1<J1; j1++)
      t+=xarr[i0*x0+i1*x1+j0*x2+j1*x3]*yarr[i0*y0+i1*y1+j0*y2+j1*y3];
  rarr[i0*r0+i1*r1]+=t;
}

template<int TIX>
__global__ void Rtensor_addEinsum_3_2_0_kernel(const float* xarr, const float* yarr, float* rarr,
  const int x0, const int x1, const int x2, const int x3, const int x4,  
  const int y0, const int y1, const int y2, const int y3, const int y4, 
  const int r0, const int r1, const int r2, 
  const int J0, const int J1){
  int i0,i1,i2;
  REP_getix<TIX>(i0,i1,i2);

  float t=0;
  for(int j0=0; j0<J0; j0++)
    for(int j1=0; j1<J1; j1++)
      t+=xarr[i0*x0+i1*x1+i2*x2+j0*x3+j1*x4]*yarr[i0*y0+i1*y1+i2*y2+j0*y3+j1*x4];
  rarr[i0*r0+i1*r1+i2*r2]+=t; 
}



  // ---------------------------------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------------------
  

namespace cnine{

  void Rtensor_addEinsum_0_0_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const cudaStream_t& stream){
    Rtensor_addEinsum_0_0_0_kernel<1><<<1,1,0,stream>>>(xarr,yarr,rarr);
  }

  void Rtensor_addEinsum_1_0_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int y0, const int r0,
    const int n0, 
    const cudaStream_t& stream){
    if(n0<=1024){
      Rtensor_addEinsum_1_0_0_kernel<1><<<1,n0,0,stream>>>(xarr,yarr,rarr,x0,y0,r0);
      return;
    }
    Rtensor_addEinsum_1_0_0_kernel<0><<<n0,1,0,stream>>>(xarr,yarr,rarr,x0,y0,r0);
  }

  void Rtensor_addEinsum_2_0_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int x1, 
    const int y0, const int y1, 
    const int r0, const int r1,
    const int n0, const int n1,
    const cudaStream_t& stream){
    if(n0*n1<=1024){
      dim3 threads(n0,n1);
      Rtensor_addEinsum_2_0_0_kernel<2><<<1,threads,0,stream>>>(xarr,yarr,rarr,x0,x1,y0,y1,r0,r1);
      return;
    }
    if(n1<=1024){
      Rtensor_addEinsum_2_0_0_kernel<1><<<n0,n1,0,stream>>>(xarr,yarr,rarr,x0,x1,y0,y1,r0,r1);
      return;
    }
    dim3 blocks(n0,n1);
    Rtensor_addEinsum_2_0_0_kernel<0><<<blocks,1,0,stream>>>(xarr,yarr,rarr,x0,x1,y0,y1,r0,r1);
  }

  void Rtensor_addEinsum_3_0_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int x1, const int x2, 
    const int y0, const int y1, const int y2,
    const int r0, const int r1, const int r2,
    const int n0, const int n1, const int n2,
    const cudaStream_t& stream){
    if(n0*n1*n2<=1024){
      dim3 threads(n0,n1,n2);
      Rtensor_addEinsum_3_0_0_kernel<3><<<1,n0*n1,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,y0,y1,y2,r0,r1,r2);
      return;
    }
    if(n1*n2<=1024){
      dim3 threads(n1,n2);
      Rtensor_addEinsum_3_0_0_kernel<2><<<n0,threads,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,y0,y1,y2,r0,r1,r2);
      return;
    }
    if(n2<=1024){
      dim3 blocks(n0,n1);
      Rtensor_addEinsum_3_0_0_kernel<1><<<blocks,n2,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,y0,y1,y2,r0,r1,r2);
      return;
    }
    dim3 blocks(n0,n1,n2);
    Rtensor_addEinsum_3_0_0_kernel<0><<<blocks,1,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,y0,y1,y2,r0,r1,r2);
  }

  
  void Rtensor_addEinsum_0_1_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, 
    const int y0, 
    const int J0, 
    const cudaStream_t& stream){
    Rtensor_addEinsum_0_1_0_kernel<1><<<1,1,0,stream>>>(xarr,yarr,rarr,x0,y0,J0);
  }
 
  void Rtensor_addEinsum_1_1_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int x1, 
    const int y0, const int y1, 
    const int r0,
    const int n0,
    const int J0, 
    const cudaStream_t& stream){
    if(n0<=1024){
      Rtensor_addEinsum_1_1_0_kernel<1><<<1,n0,0,stream>>>(xarr,yarr,rarr,x0,x1,y0,y1,r0,J0);
      return;
    }
    Rtensor_addEinsum_1_1_0_kernel<0><<<n0,1,0,stream>>>(xarr,yarr,rarr,x0,x1,y0,y1,r0,J0);
  }
 
  void Rtensor_addEinsum_2_1_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int x1, const int x2,
    const int y0, const int y1, const int y2,
    const int r0, const int r1, 
    const int n0, const int n1,
    const int J0, 
    const cudaStream_t& stream){
    if(n0*n1<=1024){
      dim3 threads(n0,n1);
      Rtensor_addEinsum_2_1_0_kernel<2><<<1,threads,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,y0,y1,y2,r0,r1,J0);
      return;
    }
    if(n1<=1024){
      Rtensor_addEinsum_2_1_0_kernel<1><<<n0,n1,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,y0,y1,y2,r0,r1,J0);
      return;
    }
    dim3 blocks(n0,n1);
    Rtensor_addEinsum_2_1_0_kernel<0><<<blocks,1,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,y0,y1,y2,r0,r1,J0);
  }
 
  void Rtensor_addEinsum_3_1_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int x1, const int x2, const int x3,
    const int y0, const int y1, const int y2, const int y3,
    const int r0, const int r1, const int r2,
    const int n0, const int n1, const int n2,
    const int J0, 
    const cudaStream_t& stream){
    if(n0*n1*n2<=1024){
      dim3 threads(n0,n1,n2);
      Rtensor_addEinsum_3_1_0_kernel<3><<<1,threads,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,y0,y1,y2,y3,r0,r1,r2,J0);
      return;
    }
    if(n1*n2<=1024){
      dim3 threads(n1,n2);
      Rtensor_addEinsum_3_1_0_kernel<2><<<n0,threads,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,y0,y1,y2,y3,r0,r1,r2,J0);
      return;
    }
    if(n2<=1024){
      dim3 blocks(n0,n1);
      Rtensor_addEinsum_3_1_0_kernel<1><<<blocks,n2,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,y0,y1,y2,y3,r0,r1,r2,J0);
      return;
    }
    dim3 blocks(n0,n1,n2);
    Rtensor_addEinsum_3_1_0_kernel<0><<<blocks,1,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,y0,y1,y2,y3,r0,r1,r2,J0);
  }

 
  void Rtensor_addEinsum_0_2_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int x1, 
    const int y0, const int y1, 
    const int J0, const int J1, 
    const cudaStream_t& stream){
    Rtensor_addEinsum_0_2_0_kernel<1><<<1,1,0,stream>>>(xarr,yarr,rarr,x0,x1,y0,y0,J0,J1);
  }
 
  void Rtensor_addEinsum_1_2_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int x1, const int x2, 
    const int y0, const int y1, const int y2,
    const int r0, 
    const int n0, 
    const int J0, const int J1, 
    const cudaStream_t& stream){
    if(n0<=1024){
      Rtensor_addEinsum_1_2_0_kernel<1><<<1,n0,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,y0,y1,y2,r0,J0,J1);
      return;
    }
    Rtensor_addEinsum_1_2_0_kernel<0><<<n0,1,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,y0,y1,y2,r0,J0,J1);
  }
 
  void Rtensor_addEinsum_2_2_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int x1, const int x2, const int x3, 
    const int y0, const int y1, const int y2, const int y3,
    const int r0, const int r1,
    const int n0, const int n1, 
    const int J0, const int J1, 
    const cudaStream_t& stream){
    if(n0*n1<=1024){
      dim3 threads(n0,n1);
      Rtensor_addEinsum_2_2_0_kernel<2><<<1,threads,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,y0,y1,y2,y3,r0,r1,J0,J1);
      return;
    }
    if(n1<=1024){
      Rtensor_addEinsum_2_2_0_kernel<1><<<n0,n1,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,y0,y1,y2,y3,r0,r1,J0,J1);
      return;
    }
    dim3 blocks(n0,n1);
    Rtensor_addEinsum_2_2_0_kernel<0><<<blocks,1,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,y0,y1,y2,y3,r0,r1,J0,J1);
  }
 
  void Rtensor_addEinsum_3_2_0_cu(const float* xarr, const float* yarr, float* rarr, 
    const int x0, const int x1, const int x2, const int x3, const int x4, 
    const int y0, const int y1, const int y2, const int y3, const int y4,
    const int r0, const int r1, const int r2,
    const int n0, const int n1, const int n2,
    const int J0, const int J1, 
    const cudaStream_t& stream){
    if(n0*n1*n2<=1024){
      dim3 threads(n0,n1,n2);
      Rtensor_addEinsum_3_2_0_kernel<3><<<1,threads,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,r0,r1,r2,J0,J1);
    }
    if(n1*n2<=1024){
      dim3 threads(n1,n2);
      Rtensor_addEinsum_3_2_0_kernel<2><<<n0,threads,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,r0,r1,r2,J0,J1);
    }
    if(n2<=1024){
      dim3 blocks(n0,n1);
      Rtensor_addEinsum_3_2_0_kernel<1><<<blocks,n2,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,r0,r1,r2,J0,J1);
    }
    dim3 blocks(n0,n1,n2);
    Rtensor_addEinsum_3_2_0_kernel<0><<<blocks,1,0,stream>>>(xarr,yarr,rarr,x0,x1,x2,x3,x4,y0,y1,y2,y3,y4,r0,r1,r2,J0,J1);
  }
 


}

#endif 
