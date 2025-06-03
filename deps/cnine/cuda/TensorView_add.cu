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


#ifndef _TensorView_add_cu
#define _TensorView_add_cu
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "TensorView.hpp"



template<typename TYPE>
__global__ void TensorView_add_kernel_t(TYPE* rarr, TYPE* xarr, int rs0, int xs0){
  rarr[threadIdx.x*rs0]+=xarr[threadIdx.x*xs0];
}

template<typename TYPE>
__global__ void TensorView_add_kernel_tt(TYPE* rarr, TYPE* xarr, int rs0, int rs1, int xs0, int xs1){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]+=xarr[threadIdx.x*xs0+threadIdx.y*xs1];
}

template<typename TYPE>
__global__ void TensorView_add_kernel_ttt(TYPE* rarr,TYPE* xarr,  int rs0, int rs1, int rs2, int xs0, int xs1, int xs2){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]+=xarr[threadIdx.x*xs0+threadIdx.y*xs1+threadIdx.z*xs2];
}

template<typename TYPE>
__global__ void TensorView_add_kernel_bt(TYPE* rarr, TYPE* xarr, int rs0, int rs1, int xs0, int xs1){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]+=xarr[blockIdx.x*xs0+threadIdx.x*xs1];
}

template<typename TYPE>
__global__ void TensorView_add_kernel_btt(TYPE* rarr, TYPE* xarr, int rs0, int rs1, int rs2, int xs0, int xs1, int xs2){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]+=xarr[blockIdx.x*xs0+threadIdx.x*xs1+threadIdx.y*xs2];
}

template<typename TYPE>
__global__ void TensorView_add_kernel_bbt(TYPE* rarr, TYPE* xarr, int rs0, int rs1, int rs2, int xs0, int xs1, int xs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]+=xarr[blockIdx.x*xs0+blockIdx.y*xs1+threadIdx.x*xs2];
}

template<typename TYPE>
__global__ void TensorView_add_kernel_bbbt(TYPE* rarr, TYPE* xarr, int rs0, int rs1, int rs2, int rs3, int xs0, int xs1, int xs2, int xs3){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2+threadIdx.x*rs3]+=xarr[blockIdx.x*xs0+blockIdx.y*xs1+blockIdx.z*xs2+threadIdx.x*xs3];
}


namespace cnine{


  template<typename TYPE>
  void TensorView_add_cu(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const cudaStream_t& stream){
    CNINE_ASSRT(r.get_dims()==x.get_dims());
    int D=r.ndims();

    if(D==1){
      if(r.get_dim(0)>1024)
	TensorView_add_kernel_bt<<<r.get_dim(0)/1024,1024,0,stream>>>(r.get_arr(),x.get_arr(),1024*r.strides[0],r.strides[0],1024*x.strides[0],x.strides[0]);
      if(r.get_dim(0)%1024>0)
	TensorView_add_kernel_t<<<1,r.get_dim(0)%1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],x.strides[0]);
    }

    if(D==2){

      if(r.get_dim(0)*r.get_dim(1)<128){
	dim3 threads(r.get_dim(0),r.get_dim(1));
	TensorView_add_kernel_tt<<<1,threads,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],x.strides[0],x.strides[1]);
	return;
      }

      if(r.get_dim(1)<=1024){
	TensorView_add_kernel_bt<<<r.get_dim(0),r.get_dim(1),0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],x.strides[0],x.strides[1]);
	return;
      }

      dim3 blocks(r.get_dim(0),r.get_dim(1)/1024);
      TensorView_add_kernel_bbt<<<blocks,1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],1024*r.strides[1],r.strides[1],x.strides[0],1024*x.strides[1],x.strides[1]);
      if(r.get_dim(1)%1024>0)
	TensorView_add_kernel_bt<<<r.get_dim(0),r.get_dim(1)%1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],x.strides[0],x.strides[1]);

    }

    if(D==3){

      if(r.get_dim(0)*r.get_dim(1)*r.get_dim(2)<128){
	dim3 threads(r.get_dim(0),r.get_dim(1),r.get_dim(2));
	TensorView_add_kernel_ttt<<<1,threads,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
	return;
      }

      if(r.get_dim(1)*r.get_dim(2)<128){
	dim3 threads(r.get_dim(1),r.get_dim(2));
	TensorView_add_kernel_btt<<<r.get_dim(0),threads,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
	return;
      }

      if(r.get_dim(2)<=1024){
	dim3 blocks(r.get_dim(0),r.get_dim(1));
	TensorView_add_kernel_bbt<<<blocks,r.get_dim(2),0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
	return;
      }

      dim3 blocks(r.get_dim(0),r.get_dim(1),r.get_dim(2)/1024);
      TensorView_add_kernel_bbbt<<<blocks,1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],1024*r.strides[2],r.strides[2],x.strides[0],x.strides[1],1024*x.strides[2],x.strides[2]);
      if(r.get_dim(2)%1024>0){
	dim3 blocks2(r.get_dim(0),r.get_dim(1));
	TensorView_add_kernel_bbt<<<blocks2,r.get_dim(2)%1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
      }

    }    

    if(D>=4){
      CNINE_UNIMPL();
    }

  }


  // Template instantiation
  template void TensorView_add_cu<float>(const TensorView<float>&, const TensorView<float>&, const cudaStream_t&);
  template void TensorView_add_cu<int>(const TensorView<int>&, const TensorView<int>&, const cudaStream_t&);
  template void TensorView_add_cu<double>(const TensorView<double>&, const TensorView<double>&, const cudaStream_t&);

}


#endif 
