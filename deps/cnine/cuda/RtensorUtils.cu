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

#ifndef _RtensorUtils_cu
#define _RtensorUtils_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>
#include <torch/types.h>

#include "Cnine_base.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "Itensor1_view.hpp"
#include "Itensor2_view.hpp"
#include "CUDAhelpers.hpp"


// ---- Rtensor Get/set ---------------------------------------------------------------------------------------


//__global__ float Rtensor_get_kernel(const float* arr){
//  return *arr;
//}

//__global__ void Rtensor_set_kernel(float* arr, const float v){
//  *arr=v;
//}

__global__ void Rtensor_inc_kernel(float* arr, const float v){
  *arr+=v;
}


// ---- Rtensor1 copy ----------------------------------------------------------------------------------------


__global__ void Rtensor_copy_kernel_t(float* rarr, const float* arr, 
  const int s0, const int rs0){
  rarr[threadIdx.x*rs0]=arr[threadIdx.x*s0];
}


// ---- Rtensor1 add -----------------------------------------------------------------------------------------


__global__ void Rtensor_add_kernel_t(float* rarr, const float* arr, 
  const int s0, const int rs0){
  rarr[threadIdx.x*rs0]+=arr[threadIdx.x*s0];
}


// ---- Rtensor2 copy ----------------------------------------------------------------------------------------


__global__ void Rtensor_copy_kernel_tt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]=arr[threadIdx.x*s0+threadIdx.y*s1];
}

__global__ void Rtensor_copy_kernel_bt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]=arr[blockIdx.x*s0+threadIdx.x*s1];
}

__global__ void Rtensor_copy_kernel_bb(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1]=arr[blockIdx.x*s0+blockIdx.y*s1];
}


// ---- Rtensor2 add -----------------------------------------------------------------------------------------


__global__ void Rtensor_add_kernel_tt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]+=arr[threadIdx.x*s0+threadIdx.y*s1];
}

__global__ void Rtensor_add_kernel_bt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]+=arr[blockIdx.x*s0+threadIdx.x*s1];
}

__global__ void Rtensor_add_kernel_bb(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1]+=arr[blockIdx.x*s0+blockIdx.y*s1];
}


__global__ void Rtensor_add_kernel_tt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1, const float c){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]+=c*arr[threadIdx.x*s0+threadIdx.y*s1];
}

__global__ void Rtensor_add_kernel_bt(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1, const float c){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]+=c*arr[blockIdx.x*s0+threadIdx.x*s1];
}

__global__ void Rtensor_add_kernel_bb(float* rarr, const float* arr, 
  const int s0, const int s1, const int rs0, const int rs1, const float c){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1]+=c*arr[blockIdx.x*s0+blockIdx.y*s1];
}


// ---- Rtensor3 copy ----------------------------------------------------------------------------------------


__global__ void Rtensor_copy_kernel_ttt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]=arr[threadIdx.x*s0+threadIdx.y*s1+threadIdx.z*s2];
}

__global__ void Rtensor_copy_kernel_btt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]=arr[blockIdx.x*s0+threadIdx.x*s1+threadIdx.y*s2];
}

__global__ void Rtensor_copy_kernel_bbt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]=arr[blockIdx.x*s0+blockIdx.y*s1+threadIdx.x*s2];
}

__global__ void Rtensor_copy_kernel_bbb(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2]=arr[blockIdx.x*s0+blockIdx.y*s1+blockIdx.z*s2];
}


__global__ void Rtensor_copy_kernel_ttt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2, const float c){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]=arr[threadIdx.x*s0+threadIdx.y*s1+threadIdx.z*s2]*c;
}

__global__ void Rtensor_copy_kernel_btt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2, const float c){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]=arr[blockIdx.x*s0+threadIdx.x*s1+threadIdx.y*s2]*c;
}

__global__ void Rtensor_copy_kernel_bbt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2, const float c){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]=arr[blockIdx.x*s0+blockIdx.y*s1+threadIdx.x*s2]*c;
}

__global__ void Rtensor_copy_kernel_bbb(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2, const float c){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2]=arr[blockIdx.x*s0+blockIdx.y*s1+blockIdx.z*s2]*c;
}


// ---- Rtensor3 add ----------------------------------------------------------------------------------------


__global__ void Rtensor_add_kernel_ttt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]+=arr[threadIdx.x*s0+threadIdx.y*s1+threadIdx.z*s2];
}

__global__ void Rtensor_add_kernel_btt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]+=arr[blockIdx.x*s0+threadIdx.x*s1+threadIdx.y*s2];
}

__global__ void Rtensor_add_kernel_bbt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]+=arr[blockIdx.x*s0+blockIdx.y*s1+threadIdx.x*s2];
}

__global__ void Rtensor_add_kernel_bbb(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2]+=arr[blockIdx.x*s0+blockIdx.y*s1+blockIdx.z*s2];
}


__global__ void Rtensor_add_kernel_ttt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2, const float c){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]+=arr[threadIdx.x*s0+threadIdx.y*s1+threadIdx.z*s2]*c;
}

__global__ void Rtensor_add_kernel_btt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2, const float c){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]+=arr[blockIdx.x*s0+threadIdx.x*s1+threadIdx.y*s2]*c;
}

__global__ void Rtensor_add_kernel_bbt(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2, const float c){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]+=arr[blockIdx.x*s0+blockIdx.y*s1+threadIdx.x*s2]*c;
}

__global__ void Rtensor_add_kernel_bbb(float* rarr, const float* arr, 
  const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2, const float c){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2]+=arr[blockIdx.x*s0+blockIdx.y*s1+blockIdx.z*s2]*c;
}


// ----------------------------------------------------------------------------------------------------------


__global__ void batched_add_kernel_0(float* rarr, const float* arr, const int sb, const int s0){
  rarr[blockIdx.x*sb+threadIdx.x*s0]+=arr[blockIdx.x*sb+threadIdx.x*s0];  
}

__global__ void batched_add_kernel_1(float* rarr, const float* arr, const int sb, const int s0, const int s1){
  rarr[blockIdx.x*sb+blockIdx.y*s0+threadIdx.x*s1]+=arr[blockIdx.x*sb+blockIdx.y*s0+threadIdx.x*s1];  
}



// ----------------------------------------------------------------------------------------------------------


__global__ void Rtensor_sum0_into_kernel_0(float* rarr, const float* arr, const int rs0, const int s0, const int s1, const int n){
  float t=0;
  for(int i=0; i<n; i++)
    t+=arr[i*s0+threadIdx.x*s1];
  rarr[threadIdx.x*rs0]+=t;
}

__global__ void Rtensor_sum0_into_kernel_0(float* rarr, const float* arr, const int rs0, const int s0, const int s1, const int n, const float c){
  float t=0;
  for(int i=0; i<n; i++)
    t+=arr[i*s0+threadIdx.x*s1];
  rarr[threadIdx.x*rs0]+=t*c;
}

__global__ void Rtensor_sum0_into_kernel_1(float* rarr, const float* arr, const int rs0, const int s0, const int s1, const int n){
  float t=0;
  for(int i=0; i<n; i++)
    t+=arr[i*s0+(blockIdx.x*1024+threadIdx.x)*s1];
  rarr[(blockIdx.x*1024+threadIdx.x)*rs0]+=t;
}

__global__ void Rtensor_sum0_into_kernel_1(float* rarr, const float* arr, const int rs0, const int s0, const int s1, const int n, const float c){
  float t=0;
  for(int i=0; i<n; i++)
    t+=arr[i*s0+(blockIdx.x*1024+threadIdx.x)*s1];
  rarr[(blockIdx.x*1024+threadIdx.x)*rs0]+=t*c;
}


__global__ void Rtensor3_sum1_into_kernel_0(float* rarr, const float* arr, const int rs0, const int rs1, 
  const int s0, const int s1, const int s2, const int n){
  float t=0;
  for(int i=0; i<n; i++)
    t+=arr[blockIdx.x*s0+i*s1+threadIdx.x*s2];
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]+=t;
}


// ----------------------------------------------------------------------------------------------------------


__global__ void ScaleSomeSlices_kernel_t(float* arr, const int s0, const int s1, const int* ix, const float* c){
  arr[ix[blockIdx.x]*s0+threadIdx.x*s1]*=c[blockIdx.x];
}

__global__ void ScaleSomeSlices_kernel_tt(float* arr, const int s0, const int s1, const int s2, const int* ix, const float* c){
  arr[ix[blockIdx.x]*s0+threadIdx.x*s1+threadIdx.y*s2]*=c[blockIdx.x];
}

__global__ void ScaleSomeSlices_kernel_bt(float* arr, const int s0, const int s1, const int s2, const int* ix, const float* c){
  arr[ix[blockIdx.x]*s0+blockIdx.y*s1+threadIdx.x*s2]*=c[blockIdx.x];
}


// ----------------------------------------------------------------------------------------------------------


__global__ void GivensSomeSlices_kernel_t(float* arr, const int s0, const int s1, const int* ix, const float* coeffs){
  int offs=threadIdx.x*s1;
  int ix0=ix[blockIdx.x*2];
  int ix1=ix[blockIdx.x*2+1];
  float c0=coeffs[blockIdx.x*2];
  float c1=coeffs[blockIdx.x*2+1];
  float t0=arr[ix0*s0+offs];
  float t1=arr[ix1*s0+offs];
  arr[ix0*s0+offs]=c0*t0+c1*t1;
  arr[ix1*s0+offs]=c0*t1-c1*t0;
}

__global__ void GivensSomeSlices_kernel_tt(float* arr, const int s0, const int s1, const int s2, const int* ix, const float* coeffs){
  int offs=threadIdx.x*s1+threadIdx.y*s2;
  int ix0=ix[blockIdx.x*2];
  int ix1=ix[blockIdx.x*2+1];
  float c0=coeffs[blockIdx.x*2];
  float c1=coeffs[blockIdx.x*2+1];
  float t0=arr[ix0*s0+offs];
  float t1=arr[ix1*s0+offs];
  arr[ix0*s0+offs]=c0*t0+c1*t1;
  arr[ix1*s0+offs]=c0*t1-c1*t0;
}

__global__ void GivensSomeSlices_kernel_bt(float* arr, const int s0, const int s1, const int s2, const int* ix, const float* coeffs){
  int offs=blockIdx.x*s1+threadIdx.x*s2;
  int ix0=ix[blockIdx.x*2];
  int ix1=ix[blockIdx.x*2+1];
  float c0=coeffs[blockIdx.x*2];
  float c1=coeffs[blockIdx.x*2+1];
  float t0=arr[ix0*s0+offs];
  float t1=arr[ix1*s0+offs];
  arr[ix0*s0+offs]=c0*t0+c1*t1;
  arr[ix1*s0+offs]=c0*t1-c1*t0;
}


// ----------------------------------------------------------------------------------------------------------


__global__ void Rtensor_add_ReLU_kernel(float* rarr, const float* xarr, const float alpha){
  float v=xarr[blockIdx.x*32+threadIdx.x];
  if(v>0) rarr[blockIdx.x*32+threadIdx.x]=v;
  else rarr[blockIdx.x*32+threadIdx.x]=alpha*v;
}

__global__ void Rtensor_add_ReLU_back_kernel(float* rarr, const float* garr, const float* xarr, const float alpha){
  float v=garr[blockIdx.x*32+threadIdx.x];
  if(xarr[blockIdx.x*32+threadIdx.x]>0) 
    rarr[blockIdx.x*32+threadIdx.x]=v;
  else 
    rarr[blockIdx.x*32+threadIdx.x]=alpha*v;
}


// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------


namespace cnine{

  
  float Rtensor_get_cu(const float* p){
    float r=0;
    CUDA_SAFE(cudaMemcpy(&r,p,sizeof(float),cudaMemcpyDeviceToHost));
    return r;
  }

  void Rtensor_set_cu(float* p, const float v){
    //Rtensor_set_kernel<<<1,1>>>(p,v);
    CUDA_SAFE(cudaMemcpy(p,&v,sizeof(float),cudaMemcpyHostToDevice));
  }

  void Rtensor_inc_cu(float* p, const float v){
    Rtensor_inc_kernel<<<1,1>>>(p,v);
  }


  void Rtensor_copy_cu(const Rtensor1_view& r, const Rtensor1_view& x, const cudaStream_t& stream){
    if(x.n0<1024){
      Rtensor_copy_kernel_t<<<0,x.n0,0,stream>>>(r.arr,x.arr,x.s0,r.s0);
      return;
    }
    Rtensor_copy_kernel_bt<<<x.n0/1024,1024,0,stream>>>(r.arr,x.arr,1024*x.s0,x.s0,1024*r.s0,r.s0);
    Rtensor_copy_kernel_t<<<0,x.n0%1024,0,stream>>>(r.arr+(x.n0-x.n0%1024)*r.s0,x.arr+(x.n0-x.n0%1024),x.s0,r.s0);
  }

  void Rtensor_copy_cu(const Rtensor2_view& r, const Rtensor2_view& x, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_copy_kernel_tt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_copy_kernel_bt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_copy_kernel_bb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);});
  }

  void Rtensor_copy_cu(const Rtensor3_view& r, const Rtensor3_view& x, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_copy_kernel_ttt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_copy_kernel_btt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_copy_kernel_bbt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_copy_kernel_bbb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);});
  }


  void Rtensor_add_cu(const Rtensor1_view& r, const Rtensor1_view& x, const cudaStream_t& stream){
    if(x.n0<1024){
      Rtensor_add_kernel_t<<<0,x.n0,0,stream>>>(r.arr,x.arr,x.s0,r.s0);
      return;
    }
    Rtensor_add_kernel_bt<<<x.n0/1024,1024,0,stream>>>(r.arr,x.arr,1024*x.s0,x.s0,1024*r.s0,r.s0);
    Rtensor_add_kernel_t<<<0,x.n0%1024,0,stream>>>(r.arr+(x.n0-x.n0%1024)*r.s0,x.arr+(x.n0-x.n0%1024),x.s0,r.s0);
    // blocks should be 1???
  }

  void Rtensor_add_cu(const Rtensor2_view& r, const Rtensor2_view& x, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_add_kernel_tt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_add_kernel_bt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_add_kernel_bb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1);});
  }

  void Rtensor_add_cu(const Rtensor2_view& r, const Rtensor2_view& x, const float c, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_add_kernel_tt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1,c);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_add_kernel_bt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1,c);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int rs0, const int rs1){      
	Rtensor_add_kernel_bb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,rs0,rs1,c);});
  }

  void Rtensor_add_cu(const Rtensor3_view& r, const Rtensor3_view& x, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_ttt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_btt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_bbt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_bbb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2);});
  }

  void Rtensor_add_cu(const Rtensor3_view& r, const Rtensor3_view& x, const float c, const cudaStream_t& stream){
    dispatch(r,x,
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_ttt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2,c);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_btt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2,c);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_bbt<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2,c);},
      [&](const dim3& blocks, const dim3& threads, const int s0, const int s1, const int s2, const int rs0, const int rs1, const int rs2){      
	Rtensor_add_kernel_bbb<<<blocks,threads,0,stream>>>(r.arr,x.arr,s0,s1,s2,rs0,rs1,rs2,c);});
  }


  void batched_add_cu(float* rarr, const float* arr, const int b, const int sb, const int n, const int s, const cudaStream_t& stream){
    if(n<=1024){
      batched_add_kernel_0<<<b,n,0,stream>>>(rarr,arr,sb,s);
    }else{
      dim3 blocks(b,n/1024);
      batched_add_kernel_1<<<blocks,1024,0,stream>>>(rarr,arr,sb,s*1024,s);
      if(n%1024>0) batched_add_kernel_0<<<b,n%1024,0,stream>>>(rarr+(n-n%1024)*s,arr+(n-n%1024)*s,sb,s);
    }
  }


  void Rtensor_sum0_into_cu(const Rtensor1_view& r, const Rtensor2_view& x, const cudaStream_t& stream){
    const int n=x.n1;
    CNINE_ASSRT(r.n0==n);
    if(n<=1024){
      Rtensor_sum0_into_kernel_0<<<1,n,0,stream>>>(r.arr,x.arr,r.s0,x.s0,x.s1,x.n0);
    }else{
      Rtensor_sum0_into_kernel_1<<<n/1024,1024,0,stream>>>(r.arr,x.arr,r.s0,x.s0,x.s1,x.n0);
      if(n%1024>0) Rtensor_sum0_into_kernel_0<<<1,n%1024,0,stream>>>(r.arr+(n-n%1024)*r.s0,x.arr+(n-n%1024)*x.s1,r.s0,x.s0,x.s1,x.n0);
    }
  }

  void Rtensor_sum0_into_cu(const Rtensor1_view& r, const Rtensor2_view& x, const float c, const cudaStream_t& stream){
    const int n=x.n1;
    CNINE_ASSRT(r.n0==n);
    if(n<=1024){
      Rtensor_sum0_into_kernel_0<<<1,n,0,stream>>>(r.arr,x.arr,r.s0,x.s0,x.s1,x.n0,c);
    }else{
      Rtensor_sum0_into_kernel_1<<<n/1024,1024,0,stream>>>(r.arr,x.arr,r.s0,x.s0,x.s1,x.n0,c);
      if(n%1024>0) Rtensor_sum0_into_kernel_0<<<1,n%1024,0,stream>>>(r.arr+(n-n%1024)*r.s0,x.arr+(n-n%1024)*x.s1,r.s0,x.s0,x.s1,x.n0,c);
    }
  }

  void Rtensor3_sum1_into_cu(const Rtensor2_view& r, const Rtensor3_view& x, const cudaStream_t& stream){
    CNINE_ASSRT(r.n0==x.n0);
    CNINE_ASSRT(r.n1==x.n2);
    if(x.n2<=1024){
      Rtensor3_sum1_into_kernel_0<<<x.n0,x.n2,0,stream>>>(r.arr,x.arr,r.s0,r.s1,x.s0,x.s1,x.s2,x.n1);
    }else{
      CNINE_UNIMPL();
    }
  }



  void ScaleSomeSlices_cu(const Rtensor2_view& x, const Itensor1_view& indices, const Rtensor1_view& coeffs, const cudaStream_t& stream){
    const int N=indices.n0;
    const int M=x.n1;
    if(M<=1024){
      ScaleSomeSlices_kernel_t<<<N,M,0,stream>>>(x.arr,x.s0,x.s1,indices.arr,coeffs.arr);
      return;
    }
    CNINE_UNIMPL();
  }

  void ScaleSomeSlices_cu(const Rtensor3_view& x, const Itensor1_view& indices, const Rtensor1_view& coeffs, const cudaStream_t& stream){
    const int N=indices.n0;

    if(x.n1*x.n2<=1024){
      dim3 threads(x.n1,x.n2);
      ScaleSomeSlices_kernel_tt<<<N,threads,0,stream>>>(x.arr,x.s0,x.s1,x.s2,indices.arr,coeffs.arr);
      return;
    }
    
    if(x.n1<=1024||x.n2<=1024){
      if(x.n1%32>x.n2%32){
	dim3 blocks(N,x.n1);
	ScaleSomeSlices_kernel_bt<<<blocks,x.n2,0,stream>>>(x.arr,x.s0,x.s1,x.s2,indices.arr,coeffs.arr);
      }else{
	dim3 blocks(N,x.n2);
	ScaleSomeSlices_kernel_bt<<<blocks,x.n1,0,stream>>>(x.arr,x.s0,x.s2,x.s1,indices.arr,coeffs.arr);
      }
    }

    CNINE_UNIMPL();
  }


  void GivensSomeSlices_cu(const Rtensor2_view& x, const Itensor2_view& indices, const Rtensor2_view& coeffs, const cudaStream_t& stream){
    const int N=indices.n0;
    const int M=x.n1;
    if(M<=1024){
      GivensSomeSlices_kernel_t<<<N,M,0,stream>>>(x.arr,x.s0,x.s1,indices.arr,coeffs.arr);
      return;
    }
    CNINE_UNIMPL();
  }

  void GivensSomeSlices_cu(const Rtensor3_view& x, const Itensor2_view& indices, const Rtensor2_view& coeffs, const cudaStream_t& stream){
    const int N=indices.n0;

    if(x.n1*x.n2<=1024){
      dim3 threads(x.n1,x.n2);
      GivensSomeSlices_kernel_tt<<<N,threads,0,stream>>>(x.arr,x.s0,x.s1,x.s2,indices.arr,coeffs.arr);
      return;
    }
    
    if(x.n1<=1024||x.n2<=1024){
      if(x.n1%32>x.n2%32){
	dim3 blocks(N,x.n1);
	GivensSomeSlices_kernel_bt<<<blocks,x.n2,0,stream>>>(x.arr,x.s0,x.s1,x.s2,indices.arr,coeffs.arr);
      }else{
	dim3 blocks(N,x.n2);
	GivensSomeSlices_kernel_bt<<<blocks,x.n1,0,stream>>>(x.arr,x.s0,x.s2,x.s1,indices.arr,coeffs.arr);
      }
    }

    CNINE_UNIMPL();
  }



  void Rtensor_add_ReLU_cu(Rtensor1_view& r, const Rtensor1_view& x, const float alpha, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(r.s0==1);
    CNINE_ASSRT(x.s0==1);
    CNINE_ASSRT(r.n0==x.n0);
    if(r.n0>=32) Rtensor_add_ReLU_kernel<<<r.n0/32,32,0,stream>>>(r.arr,x.arr,alpha);
    if(r.n0%32>0) Rtensor_add_ReLU_kernel<<<1,r.n0%32,0,stream>>>(r.arr+(r.n0/32)*32,x.arr+(r.n0/32)*32,alpha);
  }

  void Rtensor_add_ReLU_back_cu(Rtensor1_view& r, const Rtensor1_view& g, const Rtensor1_view& x, const float alpha, const cudaStream_t& stream){
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(g.dev==1);
    CNINE_ASSRT(r.s0==1);
    CNINE_ASSRT(x.s0==1);
    CNINE_ASSRT(g.s0==1);
    CNINE_ASSRT(r.n0==x.n0);
    CNINE_ASSRT(g.n0==x.n0);
    if(r.n0>=32) Rtensor_add_ReLU_back_kernel<<<r.n0/32,32,0,stream>>>(r.arr,g.arr,x.arr,alpha);
    if(r.n0%32>0) Rtensor_add_ReLU_back_kernel<<<1,r.n0%32,0,stream>>>(r.arr+(r.n0/32)*32,g.arr+(r.n0/32)*32,x.arr+(r.n0/32)*32,alpha);
  }


}

#endif 

 
