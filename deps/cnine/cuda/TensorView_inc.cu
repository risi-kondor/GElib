/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _TensorView_inc_cu
#define _TensorView_inc_cu
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "TensorView.hpp"



template<typename TYPE>
__global__ void TensorView_inc_kernel_t(TYPE* rarr, TYPE x, int rs0){
  rarr[threadIdx.x*rs0]+=x;
}

template<typename TYPE>
__global__ void TensorView_inc_kernel_tt(TYPE* rarr, TYPE x, int rs0, int rs1){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]+=x;
}

template<typename TYPE>
__global__ void TensorView_inc_kernel_ttt(TYPE* rarr, TYPE x, int rs0, int rs1, int rs2){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]+=x;
}

template<typename TYPE>
__global__ void TensorView_inc_kernel_bt(TYPE* rarr, TYPE x, int rs0, int rs1){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]+=x;
}

template<typename TYPE>
__global__ void TensorView_inc_kernel_btt(TYPE* rarr, TYPE x, int rs0, int rs1, int rs2){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]+=x;
}

template<typename TYPE>
__global__ void TensorView_inc_kernel_bbt(TYPE* rarr, TYPE x, int rs0, int rs1, int rs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]+=x;
}

template<typename TYPE>
__global__ void TensorView_inc_kernel_bbbt(TYPE* rarr, TYPE x, int rs0, int rs1, int rs2, int rs3){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2+threadIdx.x*rs3]+=x;
}


namespace cnine{


  template<typename TYPE>
  void TensorView_inc_cu(const TensorView<TYPE>& r, const TYPE x, const cudaStream_t& stream){
    int D=r.ndims();

    if(D==1){
      if(r.get_dim(0)>1024)
	TensorView_inc_kernel_bt<<<r.get_dim(0)/1024,1024,0,stream>>>(r.get_arr(),x,1024*r.strides[0],r.strides[0]);
      if(r.get_dim(0)%1024>0)
	TensorView_inc_kernel_t<<<1,r.get_dim(0)%1024,0,stream>>>(r.get_arr(),x,r.strides[0]);
    }

    if(D==2){

      if(r.get_dim(0)*r.get_dim(1)<128){
	dim3 threads(r.get_dim(0),r.get_dim(1));
	TensorView_inc_kernel_tt<<<1,threads,0,stream>>>(r.get_arr(),x,r.strides[0],r.strides[1]);
	return;
      }

      if(r.get_dim(1)<=1024){
	TensorView_inc_kernel_bt<<<r.get_dim(0),r.get_dim(1),0,stream>>>(r.get_arr(),x,r.strides[0],r.strides[1]);
	return;
      }

      dim3 blocks(r.get_dim(0),r.get_dim(1)/1024);
      TensorView_inc_kernel_bbt<<<blocks,1024,0,stream>>>(r.get_arr(),x,r.strides[0],1024*r.strides[1],r.strides[1]);
      if(r.get_dim(1)%1024>0)
	TensorView_inc_kernel_bt<<<r.get_dim(0),r.get_dim(1)%1024,0,stream>>>(r.get_arr(),x,r.strides[0],r.strides[1]);

    }

    if(D==3){

      if(r.get_dim(0)*r.get_dim(1)*r.get_dim(2)<128){
	dim3 threads(r.get_dim(0),r.get_dim(1),r.get_dim(2));
	TensorView_inc_kernel_ttt<<<1,threads,0,stream>>>(r.get_arr(),x,r.strides[0],r.strides[1],r.strides[2]);
	return;
      }

      if(r.get_dim(1)*r.get_dim(2)<128){
	dim3 threads(r.get_dim(1),r.get_dim(2));
	TensorView_inc_kernel_btt<<<r.get_dim(0),threads,0,stream>>>(r.get_arr(),x,r.strides[0],r.strides[1],r.strides[2]);
	return;
      }

      if(r.get_dim(2)<=1024){
	dim3 blocks(r.get_dim(0),r.get_dim(1));
	TensorView_inc_kernel_bbt<<<blocks,r.get_dim(2),0,stream>>>(r.get_arr(),x,r.strides[0],r.strides[1],r.strides[2]);
	return;
      }

      dim3 blocks(r.get_dim(0),r.get_dim(1),r.get_dim(2)/1024);
      TensorView_inc_kernel_bbbt<<<blocks,1024,0,stream>>>(r.get_arr(),x,r.strides[0],r.strides[1],1024*r.strides[2],r.strides[2]);
      if(r.get_dim(2)%1024>0){
	dim3 blocks2(r.get_dim(0),r.get_dim(1));
	TensorView_inc_kernel_bbt<<<blocks2,r.get_dim(2)%1024,0,stream>>>(r.get_arr(),x,r.strides[0],r.strides[1],r.strides[2]);
      }

    }    

    if(D>=4){
      CNINE_UNIMPL();
    }

  }

  //Template Instantitiation
  template void TensorView_inc_cu<float>(const TensorView<float>&r, const float x, const cudaStream_t&stream);
  template void TensorView_inc_cu<double>(const TensorView<double>&r, const double x, const cudaStream_t&stream);
  template void TensorView_inc_cu<int>(const TensorView<int>&r, const int x, const cudaStream_t&stream);
}

#endif 
