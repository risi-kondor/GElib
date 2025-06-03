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

#ifndef _LtensorBLAS_cu
#define _LtensorBLAS_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

#include "Ltensor.hpp"
#include "CUDAhelpers.hpp"


// ---- Increment --------------------------------------------------------------------------------------------


template<typename TYPE>
__global__ void Ltensor_inc_kernel_t(TYPE* rarr, int rs0, TYPE v){
  rarr[threadIdx.x*rs0]+=v;
}

template<typename TYPE>
__global__ void Ltensor_inc_kernel_tt(TYPE* rarr, int rs0, int rs1, TYPE v){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]+=v;
}

template<typename TYPE>
__global__ void Ltensor_inc_kernel_ttt(TYPE* rarr, int rs0, int rs1, int rs2, TYPE v){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]+=v;
}


template<typename TYPE>
__global__ void Ltensor_inc_kernel_bt(TYPE* rarr, int rs0, int rs1, TYPE v){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]+=v;
}

template<typename TYPE>
__global__ void Ltensor_inc_kernel_btt(TYPE* rarr, int rs0, int rs1, int rs2, TYPE v){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]+=v;
}

template<typename TYPE>
__global__ void Ltensor_inc_kernel_bttt(TYPE* rarr, int rs0, int rs1, int rs2, int rs3, TYPE v){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2+threadIdx.z*rs3]+=v;
}


template<typename TYPE>
__global__ void Ltensor_inc_kernel_bbt(TYPE* rarr, int rs0, int rs1, int rs2, TYPE v){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]+=v;
}

template<typename TYPE>
__global__ void Ltensor_inc_kernel_bbtt(TYPE* rarr, int rs0, int rs1, int rs2, int rs3, TYPE v){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2+threadIdx.y*rs3]+=v;
}

template<typename TYPE>
__global__ void Ltensor_inc_kernel_bbttt(TYPE* rarr, int rs0, int rs1, int rs2, int rs3, int rs4, TYPE v){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2+threadIdx.y*rs3+threadIdx.z*rs4]+=v;
}


template<typename TYPE>
__global__ void Ltensor_inc_kernel_bbbt(TYPE* rarr, int rs0, int rs1, int rs2, int rs3, TYPE v){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2+threadIdx.x*rs3]+=v;
}

template<typename TYPE>
__global__ void Ltensor_inc_kernel_bbbtt(TYPE* rarr, int rs0, int rs1, int rs2, int rs3, int rs4, TYPE v){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2+threadIdx.x*rs3+threadIdx.y*rs4]+=v;
}

template<typename TYPE>
__global__ void Ltensor_inc_kernel_bbbttt(TYPE* rarr, int rs0, int rs1, int rs2, int rs3, int rs4, int rs5, TYPE v){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2+threadIdx.x*rs3+threadIdx.y*rs4+threadIdx.z*rs5]+=v;
}


// ---- Copy -------------------------------------------------------------------------------------------------


template<typename TYPE>
__global__ void Ltensor_copy_kernel_t(TYPE* rarr, int rs0, int xs0){
  rarr[threadIdx.x*rs0]=xarr[threadIdx.x*xs0];
}

template<typename TYPE>
__global__ void Ltensor_copy_kernel_tt(TYPE* rarr, int rs0, int rs1, int xs0, int xs1){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]=xarr[threadIdx.x*xs0+threadIdx.y*xs1];
}

template<typename TYPE>
__global__ void Ltensor_copy_kernel_ttt(TYPE* rarr, int rs0, int rs1, int rs2, int xs0, int xs1, int xs2){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]=xarr[threadIdx.x*xs0+threadIdx.y*xs1+threadIdx.z*xs2];
}

template<typename TYPE>
__global__ void Ltensor_copy_kernel_bt(TYPE* rarr, int rs0, int rs1, int xs0, int xs1){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]=xarr[blockIdx.x*xs0+threadIdx.x*xs1];
}

template<typename TYPE>
__global__ void Ltensor_copy_kernel_btt(TYPE* rarr, int rs0, int rs1, int rs2, int xs0, int xs1, int xs2){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]=xarr[blockIdx.x*xs0+threadIdx.x*xs1+threadIdx.y*xs2];
}

template<typename TYPE>
__global__ void Ltensor_copy_kernel_bbt(TYPE* rarr, int rs0, int rs1, int rs2, int xs0, int xs1, int xs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]=xarr[blockIdx.x*xs0+blockIdx.y*xs1+threadIdx.x*xs2];
}

template<typename TYPE>
__global__ void Ltensor_copy_kernel_bbbt(TYPE* rarr, int rs0, int rs1, int rs2, int rs3, int xs0, int xs1, int xs2, int xs3){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2+threadIdx.x*rs3]=xarr[blockIdx.x*xs0+blockIdx.y*xs1+blockIdx.z*xs2+threadIdx.x*xs3];
}


// ---- Add --------------------------------------------------------------------------------------------------


template<typename TYPE>
__global__ void Ltensor_add_kernel_t(TYPE* rarr, int rs0, int xs0){
  rarr[threadIdx.x*rs0]+=xarr[threadIdx.x*xs0];
}

template<typename TYPE>
__global__ void Ltensor_add_kernel_tt(TYPE* rarr, int rs0, int rs1, int xs0, int xs1){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1]+=xarr[threadIdx.x*xs0+threadIdx.y*xs1];
}

template<typename TYPE>
__global__ void Ltensor_add_kernel_ttt(TYPE* rarr, int rs0, int rs1, int rs2, int xs0, int xs1, int xs2){
  rarr[threadIdx.x*rs0+threadIdx.y*rs1+threadIdx.z*rs2]+=xarr[threadIdx.x*xs0+threadIdx.y*xs1+threadIdx.z*xs2];
}

template<typename TYPE>
__global__ void Ltensor_add_kernel_bt(TYPE* rarr, int rs0, int rs1, int xs0, int xs1){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1]+=xarr[blockIdx.x*xs0+threadIdx.x*xs1];
}

template<typename TYPE>
__global__ void Ltensor_add_kernel_btt(TYPE* rarr, int rs0, int rs1, int rs2, int xs0, int xs1, int xs2){
  rarr[blockIdx.x*rs0+threadIdx.x*rs1+threadIdx.y*rs2]+=xarr[blockIdx.x*xs0+threadIdx.x*xs1+threadIdx.y*xs2];
}

template<typename TYPE>
__global__ void Ltensor_add_kernel_bbt(TYPE* rarr, int rs0, int rs1, int rs2, int xs0, int xs1, int xs2){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+threadIdx.x*rs2]+=xarr[blockIdx.x*xs0+blockIdx.y*xs1+threadIdx.x*xs2];
}

template<typename TYPE>
__global__ void Ltensor_add_kernel_bbbt(TYPE* rarr, int rs0, int rs1, int rs2, int rs3, int xs0, int xs1, int xs2, int xs3){
  rarr[blockIdx.x*rs0+blockIdx.y*rs1+blockIdx.z*rs2+threadIdx.x*rs3]+=xarr[blockIdx.x*xs0+blockIdx.y*xs1+blockIdx.z*xs2+threadIdx.x*xs3];
}



// -----------------------------------------------------------------------------------------------------------


namespace cnine{


  template<typename TYPE>
  void Ltensor_inc_cu(const Ltensor<TYPE>& r, const TYPE v, const cudaStream_t& stream){
    int D=r.ndims();
    if(D==1){
      if(r.dim[0]>=1024)
	Ltensor_inc_kernel_bt<<<r.dim[0]/1024,1024,0,stream>>>(r.get_arr(),1024*r.strides[0],r.strides[0],v);
      Ltensor_inc_kernel_t<<<1,r.dim[0]%1024,0,stream>>>(r.get_arr(),r.strides[0],v);
    }
    if(D==2){
      if(r.dim[0]*r.dim[1]<128){
	dim3 threads(r.dim[0],r.dim[1]);
	Ltensor_inc_kernel_tt<<<1,threads,0,stream>>>(r.get_arr(),r.strides[0],r.strides[1],v);
	return;
      }
      if(r.dim[1]<=1024){
	Ltensor_inc_kernel_bt<<<r.dim[0],R.dim[1],0,stream>>>(r.get_arr(),r.strides[0],r.strides[1],v);
	return;
      }
      dim3 blocks(r.dim[0],r.dim[1]/1024);
      Ltensor_inc_kernel_bbt<<<blocks,1024,0,stream>>>(r.get_arr(),r.strides[0],1024*r.strides[1],r.strides[1],v);
      Ltensor_inc_kernel_bt<<<R.dim[0],R.dim[1]%1024,0,stream>>>(r.get_arr(),r.strides[0],r.strides[1],v);
    }
    if(D==3){
      if(r.dim[0]*r.dim[1]*r.dim[2]<128){
	dim3 threads(R.dim[0],R.dim[1],R.dim[2]);
	Ltensor_inc_kernel_ttt<<<1,threads,0,stream>>>(r.get_arr(),r.strides[0],R.strides[1],r.strides[2],v);
	return;
      }
      if(r.dim[1]*r.dim[2]<128){
	dim3 threads(R.dim[1],R.dim[2]);
	Ltensor_inc_kernel_btt<<<R.dim[0],threads,0,stream>>>(r.get_arr(),r.strides[0],r.strides[1],r.strides[2],v);
	return;
      }
      if(r.dim[2]<=1024){
	dim3 blocks(R.dim[0],R.dim[1]);
	Ltensor_inc_kernel_bbt<<<blocks,R.dim[2],0,stream>>>(r.get_arr(),r.strides[0],r.strides[1],r.strides[2],v);
	return;
      }
      dim3 blocks(r.dim[0],r.dim[1],r.dim[2]/1024);
      Ltensor_inc_kernel_bbbt<<<blocks,1024,0,stream>>>(r.get_arr(),r.strides[0],r.strides[1],1024*r.strides[2],r.strides[2],v);
      dim3 blocks2(r.dim[0],r.dim[1]);
      Ltensor_inc_kernel_bbt<<<blocks2,r.dim[2]%1024,0,stream>>>(r.get_arr(),r.strides[0],r.strides[1],r.strides[2],v);
    }    
    if(D>=4){
      CNINE_UNIMPL();
    }
  }


  template<typename TYPE>
  void Ltensor_copy_cu(const Ltensor<TYPE>& r, const Ltensor<TYPE>& x, const cudaStream_t& stream){
    int D=r.ndims();
    if(D==1){
      if(r.dim[0]>1024)
	Ltensor_copy_kernel_bt<<<r.dim[0]/1024,1024,0,stream>>>(r.get_arr(),x.get_arr(),1024*r.strides[0],r.strides[0],1024*x.strides[0],x.strides[0]);
      Ltensor_copy_kernel_t<<<1,r.dim[0]%1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],x.strides[0]);
    }
    if(D==2){
      if(r.dim[0]*r.dim[1]<128){
	dim3 threads(R.dim[0],R.dim[1]);
	Ltensor_copy_kernel_tt<<<1,threads,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],x.strides[0],x.strides[1]);
	return;
      }
      if(R.dim[1]<=1024){
	Ltensor_copy_kernel_bt<<<r.dim[0],r.dim[1],0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],x.strides[0],x.strides[1]);
	return;
      }
      dim3 blocks(r.dim[0],r.dim[1]/1024);
      Ltensor_copy_kernel_bbt<<<blocks,1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],1024*r.strides[1],r.strides[1],x.strides[0],1024*x.strides[1],x.strides[1]);
      Ltensor_copy_kernel_bt<<<r.dim[0],r.dim[1]%1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],x.strides[0],x.strides[1]);
    }
    if(D==3){
      if(r.dim[0]*r.dim[1]*r.dim[2]<128){
	dim3 threads(r.dim[0],r.dim[1],r.dim[2]);
	Ltensor_copy_kernel_ttt<<<1,threads,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
	return;
      }
      if(r.dim[1]*r.dim[2]<128){
	dim3 threads(R.dim[1],R.dim[2]);
	Ltensor_copy_kernel_btt<<<R.dim[0],threads,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
	return;
      }
      if(r.dim[2]<=1024){
	dim3 blocks(r.dim[0],r.dim[1]);
	Ltensor_copy_kernel_bbt<<<blocks,r.dim[2],0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
	return;
      }
      dim3 blocks(r.dim[0],r.dim[1],r.dim[2]/1024);
      Ltensor_copy_kernel_bbbt<<<blocks,1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],1024*r.strides[2],r.strides[2],x.strides[0],x.strides[1],1024*x.strides[2],x.strides[2]);
      dim3 blocks2(r.dim[0],r.dim[1]);
      Ltensor_copy_kernel_bbt<<<blocks2,r.dim[2]%1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
    }    
    if(D>=4){
      CNINE_UNIMPL();
    }
  }


  template<typename TYPE>
  void Ltensor_add_cu(const Ltensor<TYPE>& r, const Ltensor<TYPE>& x, const cudaStream_t& stream){
    int D=r.ndims();
    if(D==1){
      if(r.dim[0]>1024)
	Ltensor_add_kernel_bt<<<r.dim[0]/1024,1024,0,stream>>>(r.get_arr(),x.get_arr(),1024*r.strides[0],r.strides[0],1024*x.strides[0],x.strides[0]);
      Ltensor_add_kernel_t<<<1,r.dim[0]%1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],x.strides[0]);
    }
    if(D==2){
      if(r.dim[0]*r.dim[1]<128){
	dim3 threads(R.dim[0],R.dim[1]);
	Ltensor_add_kernel_tt<<<1,threads,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],x.strides[0],x.strides[1]);
	return;
      }
      if(R.dim[1]<=1024){
	Ltensor_add_kernel_bt<<<r.dim[0],r.dim[1],0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],x.strides[0],x.strides[1]);
	return;
      }
      dim3 blocks(r.dim[0],r.dim[1]/1024);
      Ltensor_add_kernel_bbt<<<blocks,1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],1024*r.strides[1],r.strides[1],x.strides[0],1024*x.strides[1],x.strides[1]);
      Ltensor_add_kernel_bt<<<r.dim[0],r.dim[1]%1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],x.strides[0],x.strides[1]);
    }
    if(D==3){
      if(r.dim[0]*r.dim[1]*r.dim[2]<128){
	dim3 threads(r.dim[0],r.dim[1],r.dim[2]);
	Ltensor_add_kernel_ttt<<<1,threads,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
	return;
      }
      if(r.dim[1]*r.dim[2]<128){
	dim3 threads(R.dim[1],R.dim[2]);
	Ltensor_add_kernel_btt<<<R.dim[0],threads,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
	return;
      }
      if(r.dim[2]<=1024){
	dim3 blocks(r.dim[0],r.dim[1]);
	Ltensor_add_kernel_bbt<<<blocks,r.dim[2],0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
	return;
      }
      dim3 blocks(r.dim[0],r.dim[1],r.dim[2]/1024);
      Ltensor_add_kernel_bbbt<<<blocks,1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],1024*r.strides[2],r.strides[2],x.strides[0],x.strides[1],1024*x.strides[2],x.strides[2]);
      dim3 blocks2(r.dim[0],r.dim[1]);
      Ltensor_add_kernel_bbt<<<blocks2,r.dim[2]%1024,0,stream>>>(r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],x.strides[1],x.strides[2]);
    }    
    if(D>=4){
      CNINE_UNIMPL();
    }
  }


}


namespace cnine{
  
  template<> void Ltensor_inc_cu(const Ltensor<int>& r, const int v, const cudaStream_t& stream){
  template<> void Ltensor_inc_cu(const Ltensor<float>& r, const float v, const cudaStream_t& stream){
  template<> void Ltensor_inc_cu(const Ltensor<double>& r, const double v, const cudaStream_t& stream){

  template<> void Ltensor_copy_cu(const Ltensor<int>& r, const Ltensor<int>& x, const cudaStream_t& stream){
  template<> void Ltensor_copy_cu(const Ltensor<float>& r, const Ltensor<float>& x, const cudaStream_t& stream){
  template<> void Ltensor_copy_cu(const Ltensor<double>& r, const Ltensor<double>& x, const cudaStream_t& stream){

  template<> void Ltensor_add_cu(const Ltensor<int>& r, const Ltensor<int>& x, const cudaStream_t& stream){
  template<> void Ltensor_add_cu(const Ltensor<float>& r, const Ltensor<float>& x, const cudaStream_t& stream){
  template<> void Ltensor_add_cu(const Ltensor<double>& r, const Ltensor<double>& x, const cudaStream_t& stream){

}
