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

#ifndef _GELib_SPharm
#define _GElib_SPharm

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include "GPUtensor.hpp"
//#include "utils.hpp"
//#include "utils.cu"
#include "SO3part.hpp"


namespace GElib{

  __global__ void add_spharm_kernel(cnine::GPUtensor<float,5> r, 
				    const cnine::GPUtensor<float,5> M){
    extern __shared__ unsigned char _shared[]; 

    const int b0=blockIdx.x;
    const int b1=blockIdx.y;
    const int b2=blockIdx.z;
    const int L=(r.dims[3]-1)/2;
    const int t=threadIdx.x;
    const int ss=blockDim.x;
    float* arr=reinterpret_cast<float*>(_shared)+t;
    
    const float* xarr=M.arr+M.offset(b0,b1,b2,0,t);
    const float vx=*(xarr);
    const float vy=*(xarr+M.strides[3]);
    const float vz=*(xarr+2*M.strides[3]);
    const float length=sqrt(vx*vx+vy*vy+vz*vz); 
    const float len2=sqrt(vx*vx+vy*vy);
    
    float* rarr=r.arr+r.offset(b0,b1,b2,L,t);
    int rs=r.strides[3];

    if(len2==0 || std::isnan(vx/len2) || std::isnan(vy/len2)){
      (*rarr)+=sqrt(((float)(2*L+1))/(M_PI*4.0));
      return;
    }
    
    float x=vz/length;
    float xfact=sqrt(1.0-x*x);
    *arr=1;
    float* pprev;
    float* prev=arr;
    float* current=arr+1;
    for(int l=1; l<=L; l++){
      *(current+l*ss)=-(2.0*l-1.0)*xfact*(*(prev+(l-1)*ss));
      *(current+(l-1)*ss)=(2.0*l-1.0)*(*(prev+(l-1)*ss))*x;
      for(int m=0; m<l-1; m++)
	*(current+m*ss)=((2.0*l-1.0)*(*(prev+m*ss))*x-(l+m-1.0)*(*(pprev+m*ss)))/((float)(l-m));
      pprev=prev;
      prev=current;
      current+=(l+1)*ss;
    }

    thrust::complex<float> cphi(vx/len2,vy/len2);
    thrust::complex<float> phase(sqrt((2.0*L+1.0)/(M_PI*4.0)),0);
	    
    for(int m=0; m<=L; m++){
      thrust::complex<float> a=phase*(*(prev+m*ss));
      *reinterpret_cast<thrust::complex<float>*>(rarr+m*rs)+=a;
      if(m>0) *reinterpret_cast<thrust::complex<float>*>(rarr-m*rs)+=thrust::conj(a)*(1-2*(m%2));
      if(m<L) phase*=cphi/sqrt((float)(L-m)*(L+m+1));
    }
    
  }


  void addSPharm_cu(SO3part<float> r, cnine::TensorView<float> x, const cudaStream_t& stream){

    int b=r.getb();
    int l=r.getl();
    int n=r.getn();
    int dev=r.get_dev();
    GELIB_ASSRT(dev==1);
    GELIB_ASSRT(x.get_dev()==1);

    if(x.ndims()<2) GELIB_SKIP("x must be at least 2D.");
    if(x.dim(-2)!=3) GELIB_SKIP("dim(-2) of x must be 3.")

    if(x.ndims()>2){
      if(x.dims[0]==1) {x.dims[0]=b; x.strides[0]=0;} 
      else {if(x.dims[0]!=b) GELIB_SKIP("batch dimensions cannot be reconciled.");}
    }else{
      x.dims.prepend(b);
      x.strides.prepend(0);
    }

    GELIB_ASSRT(x.ndims()==r.ndims());
    GELIB_ASSRT(x.dims.chunk(1,x.ndims()-3)==r.get_gdims());

    r.canonicalize_to_5d();
    SO3part<float>::canonicalize_to_5d(x);
    cnine::GPUtensor<float,5> rv(r);
    cnine::GPUtensor<float,5> xv(x);

    if((size_t)(r.dims[0])*r.dims[1]*r.dims[2]>INT_MAX) GELIB_SKIP("product of block dimensions exceeds 2^31-1");
    if(n>1024) GELIB_SKIP("Channel dimension exceeds 1024");
    if(n*(l+1)*(l+2)/2>49152) GELIB_SKIP("Legendre factors do not fit in shared memory")

    dim3 blocks(r.dims[0],r.dims[1],r.dims[2]);
    int mem=n*(l+1)*(l+2)/2;

    add_spharm_kernel<<<blocks,n,mem,stream>>>(rv,xv);

  }

}

#endif 
