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
#include <cuda.h>
#include <cuda_runtime.h>
  
#include "TensorView.hpp"
#include "GatherMapB.hpp"

  
template<typename TYPE>
__global__ void gatherSlice_t_kernel(TYPE* rarr, const TYPE* xarr, 
  const int rs0, const int rs1, 
  const int xs0, const int xs1, 
  const int* ix){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x;
  int t0=threadIdx.x;

  const int* row=ix+ix[i+1];
  int n=ix[i+2]-ix[i+1]-1;
  int target=row[0];

  TYPE a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+t0*xs1];
  rarr[target*rs0+t0*rs1]+=a;
}


template<typename TYPE>
__global__ void gatherSlice_tt_kernel(TYPE* rarr, const TYPE* xarr, 
  const int rs0, const int rs1, const int rs2, 
  const int xs0, const int xs1, const int xs2, 
  const int* ix){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x;
  int t0=threadIdx.x;
  int t1=threadIdx.y;

  const int* row=ix+ix[i+1];
  int n=ix[i+2]-ix[i+1]-1;
  int target=row[0];

  TYPE a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+t0*xs1+t1*xs2];
  rarr[target*rs0+t0*rs1+t1*rs2]+=a;
}


template<typename TYPE>
__global__ void gatherSlice_ttt_kernel(TYPE* rarr, const TYPE* xarr, 
  const int rs0, const int rs1, const int rs2, const int rs3,  
  const int xs0, const int xs1, const int xs2, const int xs3, 
  const int* ix){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x;
  int t0=threadIdx.x;
  int t1=threadIdx.y;
  int t2=threadIdx.z;

  const int* row=ix+ix[i+1];
  int n=ix[i+2]-ix[i+1]-1;
  int target=row[0];

  TYPE a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+t0*xs1+t1*xs2+t2*xs3];
  rarr[target*rs0+t0*rs1+t1*rs2+t2*rs3]+=a;
}


template<typename TYPE>
__global__ void gatherSlice_bt_kernel(TYPE* rarr, const TYPE* xarr, 
  const int rs0, const int rs1, const int rs2,  
  const int xs0, const int xs1, const int xs2, 
  const int* ix){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x;
  int b0=blockIdx.y;
  int t0=threadIdx.x;

  const int* row=ix+ix[i+1];
  int n=ix[i+2]-ix[i+1]-1;
  int target=row[0];

  TYPE a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+b0*xs1+t0*xs2];
  rarr[target*rs0+b0*rs1+t0*rs2]+=a;
}


template<typename TYPE>
__global__ void gatherSlice_btt_kernel(TYPE* rarr, const TYPE* xarr, 
  const int rs0, const int rs1, const int rs2, const int rs3,   
  const int xs0, const int xs1, const int xs2, const int xs3, 
  const int* ix){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x;
  int b0=blockIdx.y;
  int t0=threadIdx.x;
  int t1=threadIdx.y;

  const int* row=ix+ix[i+1];
  int n=ix[i+2]-ix[i+1]-1;
  int target=row[0];

  TYPE a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+b0*xs1+t0*xs2+t1*xs3];
  rarr[target*rs0+b0*rs1+t0*rs2+t1*rs3]+=a;
}


template<typename TYPE>
__global__ void gatherSlice_bttt_kernel(TYPE* rarr, const TYPE* xarr, 
  const int rs0, const int rs1, const int rs2, const int rs3, const int rs4,    
  const int xs0, const int xs1, const int xs2, const int xs3, const int xs4, 
  const int* ix){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x;
  int b0=blockIdx.y;
  int t0=threadIdx.x;
  int t1=threadIdx.y;
  int t2=threadIdx.z;

  const int* row=ix+ix[i+1];
  int n=ix[i+2]-ix[i+1]-1;
  int target=row[0];

  TYPE a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+b0*xs1+t0*xs2+t1*xs3+t2*xs4];
  rarr[target*rs0+b0*rs1+t0*rs2+t1*rs3+t2*rs4]+=a;
}


template<typename TYPE>
__global__ void gatherSlice_bbt_kernel(TYPE* rarr, const TYPE* xarr, 
  const int rs0, const int rs1, const int rs2,  const int rs3,  
  const int xs0, const int xs1, const int xs2,  const int xs3, 
  const int* ix){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x;
  int b0=blockIdx.y;
  int b1=blockIdx.z;
  int t0=threadIdx.x;

  const int* row=ix+ix[i+1];
  int n=ix[i+2]-ix[i+1]-1;
  int target=row[0];

  TYPE a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+b0*xs1+b1*xs2+t0*xs3];
  rarr[target*rs0+b0*rs1+b1*rs2+t0*rs3]+=a;
}


template<typename TYPE>
__global__ void gatherSlice_bbtt_kernel(TYPE* rarr, const TYPE* xarr, 
  const int rs0, const int rs1, const int rs2,  const int rs3, const int rs4,  
  const int xs0, const int xs1, const int xs2,  const int xs3, const int xs4, 
  const int* ix){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x;
  int b0=blockIdx.y;
  int b1=blockIdx.z;
  int t0=threadIdx.x;
  int t1=threadIdx.y;

  const int* row=ix+ix[i+1];
  int n=ix[i+2]-ix[i+1]-1;
  int target=row[0];

  TYPE a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+b0*xs1+b1*xs2+t0*xs3+t1*xs4];
  rarr[target*rs0+b0*rs1+b1*rs2+t0*rs3+t1*xs4]+=a;
}


template<typename TYPE>
__global__ void gatherSlice_bbttt_kernel(TYPE* rarr, const TYPE* xarr, 
  const int rs0, const int rs1, const int rs2,  const int rs3, const int rs4, const int rs5,   
  const int xs0, const int xs1, const int xs2,  const int xs3, const int xs4, const int xs5, 
  const int* ix){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x;
  int b0=blockIdx.y;
  int b1=blockIdx.z;
  int t0=threadIdx.x;
  int t1=threadIdx.y;
  int t2=threadIdx.z;

  const int* row=ix+ix[i+1];
  int n=ix[i+2]-ix[i+1]-1;
  int target=row[0];

  TYPE a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+b0*xs1+b1*xs2+t0*xs3+t1*xs4+t2*xs5];
  rarr[target*rs0+b0*rs1+b1*rs2+t0*rs3+t1*xs4+t2*rs5]+=a;
}


namespace cnine{


  template<typename TYPE>
  void TensorView_gather_cu(TensorView<TYPE> r, TensorView<TYPE> x, const GatherMapB& g, const cudaStream_t& stream){

    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);

    int D=r.ndims();
    CNINE_ASSRT(x.ndims()==D);
    CNINE_ASSRT(D>=2); // make a separate case for D=1
    for(int i=1; i<D; i++)
      CNINE_ASSRT(r.dim(i)==x.dim(i));

    if(r.dim(D-1)>1024){
      r.reset(r.split(D-1,1024));
      x.reset(x.split(D-1,1024));
      D++;
    }

    vector<int> tdims;
    int total_threads=1;
    //int remaining_threads=1024;
    for(int i=0; i<D-1 && i<3; i++){
      if(x.dim(D-1-i)<(1024/total_threads)){
	tdims.push_back(x.dim(D-1-i));
	total_threads*=(x.dim(D-1-i));
      }else
	break;
    }

    int ntdims=tdims.size();
    int ngdims=D-1-ntdims;
    CNINE_ASSRT(ngdims<=2);

    if(ngdims==0){
      if(ntdims==1){
      	dim3 threads(tdims[0]);
	gatherSlice_t_kernel<<<g.size(),threads,0,stream>>>
	  (r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],
	    x.strides[0],x.strides[1],g.on_device(1).get_arr());
      }
      if(ntdims==2){
      	dim3 threads(tdims[1],tdims[0]);
	gatherSlice_tt_kernel<<<g.size(),threads,0,stream>>>
	  (r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],x.strides[0],
	    x.strides[1],x.strides[2],g.on_device(1).get_arr());
      }
      if(ntdims==3){
      	dim3 threads(tdims[2],tdims[1],tdims[0]);
	gatherSlice_ttt_kernel<<<g.size(),threads,0,stream>>>
	  (r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],r.strides[3],
	    x.strides[0],x.strides[1],x.strides[2],x.strides[3],g.on_device(1).get_arr());
      }
    }

    if(ngdims==1){
      if(ntdims==1){
	dim3 blocks(g.size(),x.dim(1));
      	dim3 threads(tdims[0]);
	gatherSlice_bt_kernel<<<blocks,threads,0,stream>>>
	  (r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],
	    x.strides[0],x.strides[1],x.strides[2],g.on_device(1).get_arr());
      }
      if(ntdims==2){
	dim3 blocks(g.size(),x.dim(1));
      	dim3 threads(tdims[1],tdims[0]);
	gatherSlice_btt_kernel<<<blocks,threads,0,stream>>>
	  (r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],r.strides[3],
	    x.strides[0],x.strides[1],x.strides[2],x.strides[3],g.on_device(1).get_arr());
      }
      if(ntdims==3){
	dim3 blocks(g.size(),x.dim(1));
      	dim3 threads(tdims[2],tdims[1],tdims[0]);
	gatherSlice_bttt_kernel<<<blocks,threads,0,stream>>>
	  (r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],r.strides[3],r.strides[4],
	    x.strides[0],x.strides[1],x.strides[2],x.strides[3],x.strides[4],g.on_device(1).get_arr());
      }
    }

    if(ngdims==2){
      if(ntdims==1){
	dim3 blocks(g.size(),x.dim(1),x.dim(2));
      	dim3 threads(tdims[0]);
	gatherSlice_bbt_kernel<<<blocks,threads,0,stream>>>
	  (r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],r.strides[3],
	    x.strides[0],x.strides[1],x.strides[2],x.strides[3],g.on_device(1).get_arr());
      }
      if(ntdims==2){
	dim3 blocks(g.size(),x.dim(1),x.dim(2));
      	dim3 threads(tdims[1],tdims[0]);
	gatherSlice_bbtt_kernel<<<blocks,threads,0,stream>>>
	  (r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],r.strides[3],r.strides[4],
	    x.strides[0],x.strides[1],x.strides[2],x.strides[3],x.strides[4],g.on_device(1).get_arr());
      }
      if(ntdims==3){
	dim3 blocks(g.size(),x.dim(1),x.dim(2));
      	dim3 threads(tdims[2],tdims[1],tdims[0]);
	gatherSlice_bbttt_kernel<<<blocks,threads,0,stream>>>
	  (r.get_arr(),x.get_arr(),r.strides[0],r.strides[1],r.strides[2],r.strides[3],r.strides[4],r.strides[5],
	    x.strides[0],x.strides[1],x.strides[2],x.strides[3],x.strides[4],x.strides[5],g.on_device(1).get_arr());
      }
    }

  }


  template void TensorView_gather_cu<int>(TensorView<int> r, TensorView<int> x, const GatherMapB& gmap, const cudaStream_t& stream);
  template void TensorView_gather_cu<float>(TensorView<float> r, TensorView<float> x, const GatherMapB& gmap, const cudaStream_t& stream);
  template void TensorView_gather_cu<double>(TensorView<double> r, TensorView<double> x, const GatherMapB& gmap, const cudaStream_t& stream);

  template void TensorView_gather_cu<complex<float> >(TensorView<complex<float> > r, TensorView<complex<float> > x, const GatherMapB& gmap, const cudaStream_t& stream);

}


#endif 
