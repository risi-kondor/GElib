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

#ifndef _gatherRows_cu
#define _gatherRows_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cnine_base.hpp"
#include "Rtensor2_view.hpp"
#include "GatherMapB.hpp"
#include "GatherMapPack.hpp"
#include "WeightedGatherMapB.hpp"
#include "minivec.hpp"
#include "AsyncGPUbuffer.hpp"


__global__ void gatherRows_kernel(float* rarr, const int rs0, const float* xarr, const int xs0, const int* ix, const int N, const int nc){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N) return;

  int t=threadIdx.y;
  if(t>=nc) return;

  const int* row=ix+2*N+ix[2*i];
  int n=ix[2*i+1]-1;
  int target=row[0];

  float a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+t];
  rarr[target*rs0+t]+=a;

}


__global__ void gatherRowsMulti_kernel(float* rarr, const int rs0, const float* xarr, const int xs0, 
  const int* sizes, int** maps, const int* out_offsets, const int* in_offsets, const int nc){
  extern __shared__ unsigned char _shared[]; 
  int b=blockIdx.x;
  int N=sizes[b];
  int i=blockIdx.y*blockDim.x+threadIdx.x;
  if(i>=N) return;

  const int* ix=maps[b];
  rarr+=out_offsets[b]*rs0;
  xarr+=in_offsets[b]*rs0;

  int t=threadIdx.y;
  if(t>=nc) return;

  const int* row=ix+2*N+ix[2*i];
  int n=ix[2*i+1]-1;
  int target=row[0];

  float a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[j+1]*xs0+t];
  rarr[target*rs0+t]+=a;

}


__global__ void gatherRowsw_kernel(float* rarr, const int rs0, const float* xarr, const int xs0, const int* ix, const int N, const int nc){
  extern __shared__ unsigned char _shared[]; 
  
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N) return;

  int t=threadIdx.y;
  if(t>=nc) return;

  const int* row=ix+2*N+ix[2*i];
  int n=(ix[2*i+1]-1)/2;
  int target=row[0];

  float a=0;
  for(int j=0; j<n; j++)
    a+=xarr[row[2*j+1]*xs0+t]*row[2*j+2];
  rarr[target*rs0+t]+=a;

}


__global__ void gatherRows_kernel(float* rarr, const int rs0, const float* xarr, const int xs0, const int* g, const int N, const int K, const int nc){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N) return;

  int t=threadIdx.y;
  if(t>=nc) return;

  const int* row=g+(K+1)*i;
  int target=row[0];

  float a=0;
  for(int j=0; j<K; j++)
    a+=xarr[row[j+1]*xs0+t];
  rarr[target*rs0+t]+=a;

}


namespace cnine{

  extern AsyncGPUbuffer<int>  GatherRowsMulti_ibuf;
  extern AsyncGPUbuffer<int*>  GatherRowsMulti_ipbuf;


  void gatherRows_cu(const Rtensor2_view& r, const Rtensor2_view& x, const GatherMapB& g, const cudaStream_t& stream){
    int nc=x.n1;
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(nc<=1024);
    CNINE_ASSRT(x.s1==1);
    CNINE_ASSRT(r.s1==1);

    int nwarps=roundup(nc,32)/32;
    int multi=32/nwarps;
    multi=1; // muti seems to make things worse!
    dim3 threads(multi,nwarps*32);
    //cout<<multi<<" "<<nwarps<<" "<<g.size()<<" "<<(g.size()-1)/multi+1<<endl;
    if(g.size()==0) return;

    gatherRows_kernel<<<(g.size()-1)/multi+1,threads,0,stream>>>
      (r.arr,r.s0,x.arr,x.s0,g.on_device(1),g.size(),nc);
    //cudaDeviceSynchronize();
  }


  void gatherRowsw_cu(const Rtensor2_view& r, const Rtensor2_view& x, const WeightedGatherMapB& g, const cudaStream_t& stream){
    int nc=r.n1;
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(x.n1==nc);
    CNINE_ASSRT(nc<=1024);
    CNINE_ASSRT(x.s1==1);
    CNINE_ASSRT(r.s1==1);

    int nwarps=roundup(nc,32)/32;
    int multi=32/nwarps;
    multi=1; // muti seems to make things worse!
    dim3 threads(multi,nwarps*32);
    //cout<<multi<<" "<<nwarps<<" "<<g.size()<<" "<<(g.size()-1)/multi+1<<endl;
    //cout<<g.arr.dir<<endl;

    gatherRowsw_kernel<<<(g.size()-1)/multi+1,threads,0,stream>>>
      (r.arr,r.s0,x.arr,x.s0,g.on_device(1),g.size(),nc);
    //cudaDeviceSynchronize();
  }


  void gatherRows_cu(const Rtensor2_view& r, const Rtensor2_view& x, const FixedkGatherMap& g, const cudaStream_t& stream){
    int nc=r.n1;
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(x.n1==nc);
    CNINE_ASSRT(nc<=1024);
    CNINE_ASSRT(x.s1==1);
    CNINE_ASSRT(r.s1==1);
    CNINE_ASSRT(g.strides(0)==g.getk()+1);
    CNINE_ASSRT(g.strides(1)==1);

    int nwarps=roundup(nc,32)/32;
    int multi=32/nwarps;
    multi=1; // muti seems to make things worse!
    dim3 threads(multi,nwarps*32);

    gatherRows_kernel<<<(g.size()-1)/multi+1,threads,0,stream>>> // changed
      (r.arr,r.s0,x.arr,x.s0,g.on_device(1),g.size(),g.getk(),nc);
  }


  void gatherRowsMulti_cu(const Rtensor2_view& r, const Rtensor2_view& x, 
    const vector<shared_ptr<const GatherMapB> >& maps, const Ltensor<int>& out_offsets, const Ltensor<int>& in_offsets,
    const cudaStream_t& stream){
    FNTRACE();

    int N=maps.size();
    int nc=x.n1;
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(nc<=1024);
    CNINE_ASSRT(x.s1==1);
    CNINE_ASSRT(r.s1==1);

    auto& int_buf=GatherRowsMulti_ibuf;
    auto& intp_buf=GatherRowsMulti_ipbuf;

    int_buf.reset(3*N+2);
    intp_buf.reset(N);

    int max_size=0;
    Ltensor<int> sizes({N},0);
    minivec<int*> map_pointers(N);
    for(int i=0; i<N; i++){
      int s=maps[i]->size();
      sizes.set(i,s);
      if(s>max_size) max_size=s;
      map_pointers.set(i,maps[i]->on_device(1));
    }
    if(max_size==0) return;

    //int_buf.push(0,sizes,stream);
    //int_buf.push(N,out_offsets,stream);
    //int_buf.push(2*N+1,in_offsets,stream);
    //intp_buf.push_minivec(0,map_pointers,stream);

    int_buf.push(0,sizes);
    int_buf.push(N,out_offsets);
    int_buf.push(2*N+1,in_offsets);
    int_buf.sync(stream);

    intp_buf.push_minivec(0,map_pointers);
    intp_buf.sync(stream);

    int nwarps=roundup(nc,32)/32;
    int multi=32/nwarps;
    multi=1;
    dim3 threads(multi,nwarps*32);
    dim3 blocks(N,(max_size-1)/multi+1);

    CUDA_SAFE(cudaDeviceSynchronize());

    gatherRowsMulti_kernel<<<blocks,threads,0,stream>>>
      (r.arr,r.s0,x.arr,x.s0,int_buf(0),intp_buf(0),int_buf(N),int_buf(2*N+1),nc);

  }

  void gatherRowsMulti_cu(const Rtensor2_view& r, const Rtensor2_view& x, 
    const GatherMapPack& maps, const cudaStream_t& stream){

    int N=maps.size();
    int nc=x.n1;
    CNINE_ASSRT(r.dev==1);
    CNINE_ASSRT(x.dev==1);
    CNINE_ASSRT(nc<=1024);
    CNINE_ASSRT(x.s1==1);
    CNINE_ASSRT(r.s1==1);

    int max_size=0;
    Ltensor<int> sizes({N},0);
    minivec<int*> map_pointers(N);
    for(int i=0; i<N; i++){
      bump(max_size,sizes.set(i,maps[i]->size());
	map_pointers.set(i,maps[i]->on_device(1));
    }
    if(max_size==0) return;

    auto& int_buf=GatherRowsMulti_ibuf;
    int_buf.reset(3*N);
    int_buf.push(0,sizes);
    int_buf.push(N,maps.out_offsets);
    int_buf.push(2*N,maps.in_offsets);
    int_buf.sync(stream);

    auto& intp_buf=GatherRowsMulti_ipbuf;
    intp_buf.reset(N);
    intp_buf.push_minivec(0,map_pointers);
    intp_buf.sync(stream);

    int nwarps=roundup(nc,32)/32;
    int multi=32/nwarps;
    multi=1;
    dim3 threads(multi,nwarps*32);
    dim3 blocks(N,(max_size-1)/multi+1);

    CUDA_SAFE(cudaDeviceSynchronize());

    gatherRowsMulti_kernel<<<blocks,threads,0,stream>>>
      (r.arr,r.s0,x.arr,x.s0,int_buf(0),intp_buf(0),int_buf(N),int_buf(2*N),nc);
  }


}

#endif 


    //    gatherRowsMulti_kernel<<<blocks,threads,0,stream>>>
    //(r.arr,r.s0,x.arr,x.s0,sizes.get_arr(),map_pointers.arr,
    //out_offsets_g.get_arr(),in_offsets_g.get_arr(),nc);

    //cudaDeviceSynchronize();

    //sizes.move_to_device(1);
    //map_pointers.move_to_device(1);

    //Ltensor<int> out_offsets_g=out_offsets.copy(1);
    //Ltensor<int> in_offsets_g=in_offsets.copy(1);

