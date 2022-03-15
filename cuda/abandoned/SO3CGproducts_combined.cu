#include <cuda.h>
#include <cuda_runtime.h>
#include "GElib_base.hpp"

__device__ __constant__ unsigned char cg_cmem[CNINE_CONST_MEM_SIZE];
#define _SO3CG_CUDA_CONCAT
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022s, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Fpart_addFproduct_back0_cu
#define _SO3Fpart_addFproduct_back0_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"


extern GElib::SO3_CGbank SO3_cgbank;


__device__ int loadg4(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=sourcec[i*s1+t*s2];
  }
  return offs;
}


__device__ int saveg4(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=sourcec[i*J+t];
  }
  return offs;
}


__device__ int loadg4c(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=-sourcec[i*s1+t*s2];
  }
  return offs;
}

/*
__device__ int saveg4c(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=-sourcec[i*J+t];
  }
  return offs;
}
*/


__global__ void SO3Fpart_addFproduct_back0_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
  const cnine::Ctensor3_view y, const int Cptr, const int conj){

  extern __shared__ unsigned char _shared[]; 
  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int b=blockIdx.x;
  const int t=threadIdx.x;

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int xn=x.n2;
  int yn=y.n2;
  int rn=r.n2;

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg4(x,xpr,b,t);

  float* ypr=xpr+((2*xn*xn-1)/32+1)*32;
  float* ypi;
  if(conj==0) ypi=ypr+loadg4(y,ypr,b,t);
  else ypi=ypr+loadg4c(y,ypr,b,t);

  float* rpr=ypr+((2*yn*yn-1)/32+1)*32;
  float* rpi=rpr+loadg4(r,rpr,b,t);

  __syncthreads();

  if(t<xn*yn){

    int i1=t/yn;
    float* _xpr=xpr+i1;
    float* _xpi=xpi+i1;
    
    int i2=t%yn;
    ypr=ypr+i2;
    ypi=ypi+i2;
    
    int i=i1+i2-l1-l2+l;
    float* _rpr=rpr+i;
    float* _rpi=rpi+i;

    if(i>=0 && i<rn){
      float c0=C_ptr[i1*yn+i2]*xn*yn/rn;
      
      for(int m1=-l1; m1<=l1; m1++){
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=C_ptr[(m1+l1)*yn+m2+l2];
	  const float y_r=ypr[yn*(m2+l2)];
	  const float y_i=ypi[yn*(m2+l2)];
	  const float g_r=_rpr[rn*(m1+m2+l)];
	  const float g_i=_rpi[rn*(m1+m2+l)];
	  //_xpr[xn*(m1+l1)]+=c0*c*(g_r*y_r+g_i*y_i);
	  //_xpi[xn*(m1+l1)]+=c0*c*(-g_r*y_i+g_i*y_r);
	  atomicAdd(_xpr+xn*(m1+l1),c0*c*(g_r*y_r+g_i*y_i));
	  atomicAdd(_xpi+xn*(m1+l1),c0*c*(-g_r*y_i+g_i*y_r));
	}
 
      }
    }
  }

  __syncthreads();
  
  saveg4(x,xpr,b,t);

}



namespace GElib{


  void SO3Fpart_addFproduct_back0_cu(const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& g, const cnine::Ctensor3_view& y, 
    const int conj, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(g.n1-1)/2;

    const int b=g.n0;
    assert(x.n0==b);
    assert(y.n0==b);

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;

    int nlines=cnine::roundup(x.n1*x.n2*2,32)/32+
      cnine::roundup(y.n1*y.n2*2,32)/32+
      cnine::roundup(g.n1*g.n2*2,32)/32;


    if(nlines<=384){

      SO3Fpart_addFproduct_back0_kernel<<<b,cnine::roundup(x.n2*y.n2,32),nlines*128,stream>>>
	(g,x,y,Cptr,conj);

    }else{
      cout<<"error"<<endl;
    }

  }    


}


#endif 

// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022s, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Fpart_addFproduct_back1_cu
#define _SO3Fpart_addFproduct_back1_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"


extern GElib::SO3_CGbank SO3_cgbank;


__device__ int loadg5(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=sourcec[i*s1+t*s2];
  }
  return offs;
}


__device__ int saveg5(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=sourcec[i*J+t];
  }
  return offs;
}


__device__ int loadg5c(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=-sourcec[i*s1+t*s2];
  }
  return offs;
}


__device__ int saveg5c(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=-sourcec[i*J+t];
  }
  return offs;
}



__global__ void SO3Fpart_addFproduct_back1_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
  const cnine::Ctensor3_view y, const int Cptr, const int conj){

  extern __shared__ unsigned char _shared[]; 
  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int b=blockIdx.x;
  const int t=threadIdx.x;

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int xn=x.n2;
  int yn=y.n2;
  int rn=r.n2;

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg5(x,xpr,b,t);

  float* ypr=xpr+((2*xn*xn-1)/32+1)*32;
  float* ypi;
  if(conj==0) ypi=ypr+loadg5(y,ypr,b,t);
  else ypi=ypr+loadg5c(y,ypr,b,t);

  float* rpr=ypr+((2*yn*yn-1)/32+1)*32;
  float* rpi=rpr+loadg5(r,rpr,b,t);

  __syncthreads();

  if(t<xn*yn){

    int i1=t/yn;
    float* _xpr=xpr+i1;
    float* _xpi=xpi+i1;
    
    int i2=t%yn;
    float* _ypr=ypr+i2;
    float* _ypi=ypi+i2;
    
    int i=i1+i2-l1-l2+l;
    float* _rpr=rpr+i;
    float* _rpi=rpi+i;

    if(i>=0 && i<rn){
      float c0=C_ptr[i1*yn+i2]*xn*yn/rn;
      
      for(int m1=-l1; m1<=l1; m1++){
	const float x_r=_xpr[xn*(m1+l1)];
	const float x_i=_xpi[xn*(m1+l1)];
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=C_ptr[(m1+l1)*yn+m2+l2];
	  const float g_r=_rpr[rn*(m1+m2+l)];
	  const float g_i=_rpi[rn*(m1+m2+l)];
	  //_ypr[yn*(m2+l2)]+=c*(g_r*x_r+g_i*x_i);
	  //_ypi[yn*(m2+l2)]+=c*(-g_r*x_i+g_i*x_r);
	  atomicAdd(_ypr+yn*(m2+l2),c0*c*(g_r*x_r+g_i*x_i));
	  atomicAdd(_ypi+yn*(m2+l2),c0*c*(-g_r*x_i+g_i*x_r));
	}
 
      }
    }
  }

  __syncthreads();
  
  if(conj==0) saveg5(y,ypr,b,t);
  else saveg5c(y,ypr,b,t);

}



namespace GElib{


  void SO3Fpart_addFproduct_back1_cu(const cnine::Ctensor3_view& y, const cnine::Ctensor3_view& g, const cnine::Ctensor3_view& x, 
    const int conj, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(g.n1-1)/2;

    const int b=g.n0;
    assert(x.n0==b);
    assert(y.n0==b);

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;

    int nlines=cnine::roundup(x.n1*x.n2*2,32)/32+
      cnine::roundup(y.n1*y.n2*2,32)/32+
      cnine::roundup(g.n1*g.n2*2,32)/32;


    if(nlines<=384){

      SO3Fpart_addFproduct_back1_kernel<<<b,cnine::roundup(x.n2*y.n2,32),nlines*128,stream>>>
	(g,x,y,Cptr,conj);

    }else{
      cout<<"error"<<endl;
    }

  }    


}


#endif 

// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022s, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3Fpart_addFproduct_cu
#define _SO3Fpart_addFproduct_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor2_view.hpp"
#include "Ctensor3_view.hpp"

//__device__ __constant__ unsigned char cg_cmem[32276]; 

extern GElib::SO3_CGbank SO3_cgbank;




__device__ int loadg3(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=sourcec[i*s1+t*s2];
  }
  return offs;
}


__device__ int saveg3(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=sourcec[i*J+t];
  }
  return offs;
}

__device__ int loadg3c(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=-sourcec[i*s1+t*s2];
  }
  return offs;
}


/*
__device__ int saveg3c(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=-sourcec[i*J+t];
  }
  return offs;
}
*/


__global__ void SO3Fpart_addFproduct_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
  const cnine::Ctensor3_view y, const int Cptr, const int conj){

  extern __shared__ unsigned char _shared[]; 
  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int b=blockIdx.x;
  const int t=threadIdx.x;

//printf("%d",t);

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int xn=x.n2;
  int yn=y.n2;
  int rn=r.n2;

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg3(x,xpr,b,t);

  float* ypr=xpr+((2*xn*xn-1)/32+1)*32;
  float* ypi;
  if(conj==0) ypi=ypr+loadg3(y,ypr,b,t);
  else ypi=ypr+loadg3c(y,ypr,b,t);

  float* rpr=ypr+((2*yn*yn-1)/32+1)*32;
  float* rpi=rpr+loadg3(r,rpr,b,t);

  __syncthreads();

  if(t<xn*yn){

    int i1=t/yn;
    xpr=xpr+i1;
    xpi=xpi+i1;
    
    int i2=t%yn;
    ypr=ypr+i2;
    ypi=ypi+i2;
    
    int i=i1+i2-l1-l2+l;
    float* _rpr=rpr+i;
    float* _rpi=rpi+i;

    if(i>=0 && i<rn){

      float c0=C_ptr[i1*yn+i2]*xn*yn/rn;
      
      for(int m1=-l1; m1<=l1; m1++){
	const float x_r=xpr[xn*(m1+l1)];
	const float x_i=xpi[xn*(m1+l1)];
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=C_ptr[(m1+l1)*yn+m2+l2];
	  const float y_r=ypr[yn*(m2+l2)];
	  const float y_i=ypi[yn*(m2+l2)];
	//  _rpr[rn*(m1+m2+l)]+=c0*c*(x_r*y_r-x_i*y_i); 
	  //_rpi[rn*(m1+m2+l)]+=c0*c*(x_r*y_i+x_i*y_r);
	 atomicAdd(_rpr+rn*(m1+m2+l),c0*c*(x_r*y_r-x_i*y_i)); 
	 atomicAdd(_rpi+rn*(m1+m2+l),c0*c*(x_r*y_i+x_i*y_r));
	}
 
      }
    }
  }

  __syncthreads();
  
  saveg3(r,rpr,b,t);

}



namespace GElib{


  void SO3Fpart_addFproduct_cu(const cnine::Ctensor3_view& r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int conj,const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(r.n1-1)/2;

    const int b=r.n0;
    assert(x.n0==b);
    assert(y.n0==b);

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;

    int nlines=cnine::roundup(x.n1*x.n2*2,32)/32+
      cnine::roundup(y.n1*y.n2*2,32)/32+
      cnine::roundup(r.n1*r.n2*2,32)/32;


    if(nlines<=384){

      SO3Fpart_addFproduct_kernel<<<b,cnine::roundup(x.n2*y.n2,32),nlines*128,stream>>>
	(r,x,y,Cptr,conj);

    }else{
      cout<<"error"<<endl;
    }

  }    


}


#endif 


// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partA_CGproduct_cu
#define _SO3partA_CGproduct_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

//__device__ __constant__ unsigned char cg_cmem[32276]; 


#include "SO3partArrayA.hpp"
#include "SO3_CGbank.hpp"

#include "CellwiseBinaryCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "InnerCmap.hpp"
#include "OuterCmap.hpp"
#include "MVprodCmap.hpp"
#include "VMprodCmap.hpp"
//#include "convolve1_cmap.hpp"
#include "Convolve2Cmap.hpp"

extern GElib::SO3_CGbank SO3_cgbank;


__device__ void SO3part_load_lines(float* dest, const float* source, const int nlines, const int t){
  if(t<32){
    for(int i=0; i<nlines; i++)
      dest[i*32+t]=source[i*32+t];
  }
}

__device__ void SO3part_save_lines(const float* source, float* dest, const int nlines, const int t){
  if(t<32){
    for(int i=0; i<nlines; i++)
      dest[i*32+t]=source[i*32+t];
  }
}


// ---- CGproduct --------------------------------------------------------------------------------------------


template<typename IMAP>
__global__ void SO3partA_CGproduct_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const IMAP cmap, 
  const int xn, const int yn, const int rn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int r=2*l+1;
  const int r1=2*l1+1;
  const int r2=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int rwidth=xn*yn*nch;
  const int global_rwidth=rn*nch;
  
  const int rlines=((r*rwidth-1)/32+1);
  const int xlines=((r1*xwidth-1)/32+1);
  const int ylines=((r2*ywidth-1)/32+1);

  const int rptr=0;
  const int xptr=rptr+rlines*64;
  const int yptr=xptr+xlines*64;

  int rix,xix,yix;
  int nsum;
  int lst;


  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    rix=thrust::get<0>(T);
    xix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
    //if(t==0) printf("foop1\n");
  }else{
    rix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }
  
  if(mode==1){
    if(t<32){
      for(int i=0; i<2*rlines; i++)
	shared[rptr+i*32+t]=0;
    }
  }else{
    if(t<rwidth){
      for(int i=0; i<r; i++)
	shared[rptr+i*rwidth+t]=rarr[rix*rstride+_offs+i*global_rwidth+t];
      for(int i=0; i<r; i++)
	shared[rptr+rlines*32+i*rwidth+t]=rarrc[rix*rstride+_offs+i*global_rwidth+t];
    }
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      xix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);

    //if(t==0) printf("foop3\n");
      
      __syncthreads();

      const int rpr=rptr+t;
      const int rpi=rpr+rlines*32;

      const int xcol=t/yn;
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;

      const int ycol=t%ywidth;
      const int ypr=yptr+ycol;
      const int ypi=ypr+ylines*32;


      if(t<rwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*r2+m2+l2];
	    const float y_r=shared[ypr+ywidth*(m2+l2)];
	    const float y_i=shared[ypi+ywidth*(m2+l2)];
	    shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	    shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
	  }
	}
      }

    //if(t==0) printf("foop4\n");

      __syncthreads();
  }

  //if(t==0) printf("fooq\n");
  
  if(t<rwidth){
    for(int i=0; i<r; i++)
      rarr[rix*rstride+_offs+i*global_rwidth+t]=shared[rptr+i*rwidth+t];
    for(int i=0; i<r; i++)
      rarrc[rix*rstride+_offs+i*global_rwidth+t]=shared[rptr+rlines*32+i*rwidth+t];
  }    
 
}



template<typename IMAP>
__global__ void SO3partA_CGproduct_kernel_L(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const IMAP cmap, 
  const int xn, const int yn, const int rn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int r=2*l+1;
  const int r1=2*l1+1;
  const int r2=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int rwidth=xn*nch; 
  const int global_rwidth=rn*nch;
  
  const int rlines=((r*rwidth-1)/32+1);
  const int xlines=((r1*xwidth-1)/32+1);
  const int ylines=((r2*1-1)/32+1);

  const int rptr=0;
  const int xptr=rptr+rlines*64;
  const int yptr=xptr+xlines*64;

  int rix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    rix=thrust::get<0>(T);
    xix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
  }else{
    rix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }
  

  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      xix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);


    for(int ycol=0; ycol<ywidth; ycol++){

      if(t==0){
	for(int i=0; i<r; i++)
	  shared[yptr+i]=yarr[yix*ystride+i*ywidth+ycol];
	for(int i=0; i<r; i++)
	  shared[yptr+ylines*32+i]=yarrc[yix*ystride+i*ywidth+ycol];
      }

      if(t<32){
	for(int i=0; i<2*rlines; i++)
	  shared[rptr+i*32+t]=0;
      }
      
      __syncthreads();

      const int rpr=rptr+t;
      const int rpi=rpr+rlines*32;

      const int xcol=t;
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;

      //const int ycol=t%ywidth;
      const int ypr=yptr;// +ycol;
      const int ypi=ypr+ylines*32;


      if(t<rwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*r2+m2+l2];
	    const float y_r=shared[ypr+1*(m2+l2)];
	    const float y_i=shared[ypi+1*(m2+l2)];
	    shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	    shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
	  }
	}
      }

      //if(t==0) printf("foop4\n");

      __syncthreads();

      //if(t==0) printf("fooq\n");
  
      if(t<rwidth){
	for(int i=0; i<r; i++)
	  rarr[rix*rstride+_offs+i*global_rwidth+t*ywidth+ycol]+=shared[rptr+i*rwidth+t];
	for(int i=0; i<r; i++)
	  rarrc[rix*rstride+_offs+i*global_rwidth+t*ywidth+ycol]+=shared[rptr+rlines*32+i*rwidth+t];
      }    

      __syncthreads();

    } // ycol

  } //nsum
}




// ---- back0 ------------------------------------------------------------------------------------------------


template<typename IMAP>
__global__ void SO3partA_CGproduct_back0_kernel(float* xarr, float* xarrc, float* garr, float* garrc, 
  float* yarr, float* yarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int xptr=0;
  const int gptr=xptr+xlines*64;
  const int yptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    xix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
  }else{
    xix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*xlines; i++){
	shared[xptr+i*32+t]=0;
      }
    }
  }else{
    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);

    __syncthreads();

      //const int xcol=t;
    const int xpr=xptr+t;
    const int xpi=xpr+xlines*32;
    
    for(int ycol=0; ycol<ywidth; ycol++){
      
      const int ypr=yptr+ycol;
      const int ypi=ypr+ylines*32;
      
      const int gpr=gptr+ywidth*t+ycol;
      const int gpi=gpr+glines*32;

      if(t<xwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    const float y_r=shared[ypr+ywidth*(m2+l2)];
	    const float y_i=shared[ypi+ywidth*(m2+l2)];
	    const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    shared[xpr+xwidth*(m1+l1)]+=c*(g_r*y_r+g_i*y_i); 
	    shared[xpi+xwidth*(m1+l1)]+=c*(-g_r*y_i+g_i*y_r);
	  }
	}
      }
      __syncthreads();
	
    }

  }
  
  SO3part_save_lines(shared+xptr,xarr+xix*xstride,xlines,t);
  SO3part_save_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
    
  __syncthreads();

}


template<typename IMAP>
__global__ void SO3partA_CGproduct_back0_kernel_big(float* xarr, float* xarrc, float* garr, float* garrc, 
  float* yarr, float* yarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  //const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  //const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  //const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int xptr=0;
  const int yptr=xptr+xlines*64;
  //const int yptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    xix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
  }else{
    xix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*xlines; i++){
	shared[xptr+i*32+t]=0;
      }
    }
  }else{
    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    //for(int i=0; i<rg; i++)
    //if(t<gwidth)
    //shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    //for(int i=0; i<rg; i++)
    //if(t<gwidth)
    //shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);

    __syncthreads();

      //const int xcol=t;
    const int xpr=xptr+t;
    const int xpi=xpr+xlines*32;
    
    for(int ycol=0; ycol<ywidth; ycol++){
      
      const int ypr=yptr+ycol;
      const int ypi=ypr+ylines*32;
      
      //const int gpr=gptr+ywidth*t+ycol;
      //const int gpi=gpr+glines*32;

      if(t<xwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    const float y_r=shared[ypr+ywidth*(m2+l2)];
	    const float y_i=shared[ypi+ywidth*(m2+l2)];
	    //const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    //const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    const float g_r=garr[gix*gstride+_offs+ywidth*t+ycol+(m1+m2+l)*global_gwidth];
	    const float g_i=garrc[gix*gstride+_offs+ywidth*t+ycol+(m1+m2+l)*global_gwidth];
	    shared[xpr+xwidth*(m1+l1)]+=c*(g_r*y_r+g_i*y_i); 
	    shared[xpi+xwidth*(m1+l1)]+=c*(-g_r*y_i+g_i*y_r);
	  }
	}
      }
      __syncthreads();
	
    }

  }
  
  SO3part_save_lines(shared+xptr,xarr+xix*xstride,xlines,t);
  SO3part_save_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
    
  __syncthreads();

}


// ---- back1 ------------------------------------------------------------------------------------------------


template<typename IMAP>
__global__ void SO3partA_CGproduct_back1_kernel(float* yarr, float* yarrc, float* garr, float* garrc, 
  float* xarr, float* xarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int yptr=0;
  const int gptr=yptr+ylines*64;
  const int xptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    yix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    xix=thrust::get<2>(T);
    nsum=1;
  }else{
    yix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*ylines; i++)
	shared[yptr+i*32+t]=0;
    }
  }else{
    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      xix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);

    __syncthreads();

    //const int ycol=t;
    const int ypr=yptr+t;
    const int ypi=ypr+ylines*32;
    
    for(int xcol=0; xcol<xwidth; xcol++){
      
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;
      
      const int gpr=gptr+ywidth*xcol+t;
      const int gpi=gpr+glines*32;

      if(t<ywidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    shared[ypr+ywidth*(m2+l2)]+=c*(g_r*x_r+g_i*x_i); 
	    shared[ypi+ywidth*(m2+l2)]+=c*(-g_r*x_i+g_i*x_r);
	  }
	}
      }
      __syncthreads();

    }

  }
  
  SO3part_save_lines(shared+yptr,yarr+yix*ystride,ylines,t);
  SO3part_save_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
    
  __syncthreads();

}


template<typename IMAP>
__global__ void SO3partA_CGproduct_back1_kernel_big(float* yarr, float* yarrc, float* garr, float* garrc, 
  float* xarr, float* xarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  //const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  //const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  //const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int yptr=0;
  const int xptr=yptr+ylines*64;
  //const int xptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    yix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    xix=thrust::get<2>(T);
    nsum=1;
  }else{
    yix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*ylines; i++)
	shared[yptr+i*32+t]=0;
    }
  }else{
    SO3part_load_lines(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      xix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    //for(int i=0; i<rg; i++)
    //if(t<gwidth)
    //shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    //for(int i=0; i<rg; i++)
    //if(t<gwidth)
    //shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);

    __syncthreads();

    //const int ycol=t;
    const int ypr=yptr+t;
    const int ypi=ypr+ylines*32;
    
    for(int xcol=0; xcol<xwidth; xcol++){
      
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;
      
      //const int gpr=gptr+ywidth*xcol+t;
      //const int gpi=gpr+glines*32;

      if(t<ywidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    //const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    //const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    const float g_r=garr[gix*gstride+_offs+ywidth*xcol+t+(m1+m2+l)*global_gwidth];
	    const float g_i=garrc[gix*gstride+_offs+ywidth*xcol+t+(m1+m2+l)*global_gwidth];
	    shared[ypr+ywidth*(m2+l2)]+=c*(g_r*x_r+g_i*x_i); 
	    shared[ypi+ywidth*(m2+l2)]+=c*(-g_r*x_i+g_i*x_r);
	  }
	}
      }
      __syncthreads();

    }

  }
  
  SO3part_save_lines(shared+yptr,yarr+yix*ystride,ylines,t);
  SO3part_save_lines(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
    
  __syncthreads();

}


// -----------------------------------------------------------------------------------------------------------


namespace GElib{


  template<typename CMAP>
  void SO3partA_CGproduct_cu(const CMAP& map, SO3partArrayA& r, const SO3partArrayA& x, 
    const SO3partArrayA& y, const cudaStream_t& stream, const int offs, const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=r.getl();
    const int _nch=1;
    assert(x.nbu==r.nbu);
    assert(y.nbu==r.nbu);
    int _nbu=1; if(_nbu<0) _nbu=1;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    //int nlines=x.cellstride/16+y.cellstride/16+r.cellstride/16; // should be smaller than this!
    int nlines=x.cellstride/16+y.cellstride/16+cnine::roundup(x.getn()*y.getn()*_nch*(2*l+1),32)/16;
    // nlines/=_nbu;

    //cout<<"nlines="<<nlines<<endl;

    if(nlines<=0*384){

      SO3partA_CGproduct_kernel<<<map.blockdims(),cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	  r.cellstride,x.cellstride,y.cellstride,map,
	  x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);

    }else{
      
      int nlines=x.cellstride/16+cnine::roundup(_nch*(2*yl+1),32)/16+cnine::roundup(x.getn()*_nch*(2*l+1),32)/16;

      cout<<"GElib: large CGproduct"<<endl; 

      if(nlines>384){
	cout<<"GElib error: CGproduct too big for shared memory"<<endl;
      }else{
	SO3partA_CGproduct_kernel_L<<<map.blockdims(),cnine::roundup(x.getn(),32),nlines*128,stream>>>
	  (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	    r.cellstride,x.cellstride,y.cellstride,map,
	    x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      }
    }

  }

  
  void SO3partA_CGproduct_cu(SO3partA& r, const SO3partA& x, const SO3partA& y,  const int offs, 
    const cudaStream_t& stream,const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=r.getl();
    const int _nch=1;
    assert(x.nbu==r.nbu);
    assert(y.nbu==r.nbu);
    int _nbu=1; if(_nbu<0) _nbu=1;
    cnine::CellwiseBinaryCmap map;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=cnine::roundup(x.memsize,32)/32+cnine::roundup(y.memsize,32)/32+
      cnine::roundup(x.getn()*y.getn()*_nch*(2*l+1),32)/16;

    //cout<<"nlines="<<nlines<<endl;

    if(nlines<=384){

      SO3partA_CGproduct_kernel<<<map.blockdims(),cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	  0,0,0,map,
	  x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);

    }else{
      
      int nlines=cnine::roundup(x.memsize,32)/32+cnine::roundup(y.memsize,32)/32+
        cnine::roundup(x.getn()*_nch*(2*l+1),32)/16;

      cout<<"GElib: large CGproduct"<<endl; 

      if(nlines>384){
	cout<<"GElib error: CGproduct too big for shared memory"<<endl;
      }else{
	SO3partA_CGproduct_kernel_L<<<map.blockdims(),cnine::roundup(x.getn(),32),nlines*128,stream>>>
	  (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	    0,0,0,map,
	    x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      }
    }

  }

  
  template<typename CMAP>
  void SO3partA_CGproduct_back0_cu(const CMAP& map, SO3partArrayA& x, const SO3partArrayA& g, 
    const SO3partArrayA& y, const cudaStream_t& stream, const int offs, const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=g.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+g.cellstride/16;
    assert(x.nbu==g.nbu);
    assert(y.nbu==g.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    nlines/=_nbu;

    //cout<<"nlines="<<nlines<<endl;

    if(nlines<=0*384){

      SO3partA_CGproduct_back0_kernel<<<map.blockdims(),cnine::roundup(x.getn(),32),nlines*128,stream>>>
	(x.arrg,x.arrgc,g.arrg,g.arrgc,y.arrg,y.arrgc,
	  x.cellstride,y.cellstride,g.cellstride,map,
	  x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      
    }else{

      int nlines=x.cellstride/16+y.cellstride/16;
      
      cout<<"GElib: large CGproduct_back0"<<endl; 

      if(nlines>384){
	cout<<"GElib error: CGproduct too big for shared memory"<<endl;
      }else{
	SO3partA_CGproduct_back0_kernel_big<<<map.blockdims(),cnine::roundup(std::max(x.getn(),y.getn()),32),nlines*128,stream>>>
	  (x.arrg,x.arrgc,g.arrg,g.arrgc,y.arrg,y.arrgc,
	    x.cellstride,y.cellstride,g.cellstride,map,
	    x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      }

    }

  }
  

  template<typename CMAP>
  void SO3partA_CGproduct_back1_cu(const CMAP& map, SO3partArrayA& y, const SO3partArrayA& g, 
    const SO3partArrayA& x, const cudaStream_t& stream, const int offs, const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=g.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+g.cellstride/16;
    assert(x.nbu==g.nbu);
    assert(y.nbu==g.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    nlines/=_nbu;

    //cout<<"nlines="<<nlines<<endl;

    if(nlines<=0*384){

      SO3partA_CGproduct_back1_kernel<<<map.blockdims(),cnine::roundup(y.getn(),32),nlines*128,stream>>>
	(y.arrg,y.arrgc,g.arrg,g.arrgc,x.arrg,x.arrgc,
	  x.cellstride,y.cellstride,g.cellstride,map,
	  x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      
    }else{

      int nlines=x.cellstride/16+y.cellstride/16;
      
      cout<<"GElib: large CGproduct_back1"<<endl; 

      if(nlines>384){
	cout<<"GElib error: CGproduct too big for shared memory"<<endl;
      }else{
	SO3partA_CGproduct_back1_kernel_big<<<map.blockdims(),cnine::roundup(std::max(x.getn(),y.getn()),32),nlines*128,stream>>>
	  (y.arrg,y.arrgc,g.arrg,g.arrgc,x.arrg,x.arrgc,
	    x.cellstride,y.cellstride,g.cellstride,map,
	    x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      }

    }

  }


  template void SO3partA_CGproduct_cu(const cnine::CellwiseBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::BroadcastBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::OuterCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::InnerCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::MVprodCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_cu(const cnine::Convolve2Cmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);



  template void SO3partA_CGproduct_back0_cu(const cnine::CellwiseBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_back0_cu(const cnine::BroadcastBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_back0_cu(const cnine::OuterCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);


  template void SO3partA_CGproduct_back1_cu(const cnine::CellwiseBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_back1_cu(const cnine::BroadcastBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_CGproduct_back1_cu(const cnine::OuterCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);




}

#endif 





  /*
  void SO3partA_CGproduct_cu(SO3partArrayA& r, const SO3partArrayA& x, const SO3partArrayA& y, 
    const int mode, const cudaStream_t& stream, const int offs){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=r.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+r.cellstride/16;
    assert(x.nbu==r.nbu);
    assert(y.nbu==r.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    nlines/=_nbu;

    if(mode==0){
      dim3 blocks(r.aasize,1,1);
      cnine::CellwiseImap imap;
      SO3partA_CGproduct_kernel<<<blocks,cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	  r.cellstride,x.cellstride,y.cellstride,imap,
	  x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr); // 
    }

    if(mode==1){
      dim3 blocks(x.aasize,y.aasize,1);
      cnine::OuterImap imap(r.adims[1]);
      SO3partA_CGproduct_kernel<<<blocks,cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	  r.cellstride,x.cellstride,y.cellstride,imap,
	  x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr);
    }
    

  }
    */


  /*
  void SO3partA_CGproduct_cu(SO3partArrayA& r, const SO3partArrayA& x, const SO3partArrayA& y, 
    const int rN, const int xN, const int yN, 
    const int ris, const int rjs, const int rks, 
    const int xis, const int xjs, const int xks, 
    const int yis, const int yjs, const int yks, 
    const cudaStream_t& stream, const int offs){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=r.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+r.cellstride/16;
    assert(x.nbu==r.nbu);
    assert(y.nbu==r.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    dim3 blocks(rN,xN,yN);
    nlines/=_nbu;


    SO3partA_CGproduct_kernel<<<blocks,cnine::roundup(x.getn()*y.getn(),32),nlines*128,stream>>>
      (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	ris*r.cellstride,rjs*r.cellstride, rks*r.cellstride,
	xis*x.cellstride,xjs*x.cellstride, xks*x.cellstride,
	yis*y.cellstride,yjs*y.cellstride, yks*y.cellstride,
	x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr); 
  }
  */

/*
__global__ void SO3partA_CGproduct_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, 
  const int ristride, const int xistride, const int yistride,   
  const int rjstride, const int xjstride, const int yjstride,   
  const int rkstride, const int xkstride, const int ykstride,   
  const int xfrags, const int yfrags, const int rfrags,  
  const int l1, const int l2, const int l, const int _offs, const int nch, const int Cptr){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;

  const int iix=blockIdx.x;
  const int jix=blockIdx.y;
  const int kix=blockIdx.z;

  const int t=threadIdx.x;

  const int r1=2*l1+1;
  const int r2=2*l2+1;
  const int r=2*l+1;

  const int xwidth=xfrags*nch; 
  const int ywidth=yfrags*nch; 
  const int rwidth=xfrags*yfrags*nch;
  const int global_rwidth=rfrags*nch;

  int offs=0;

  int xptr=32*offs;
  SO3part_load(offs,shared,xarr,xarrc,l1,xwidth,iix*xistride+jix*xjstride+kix*xkstride,t);

  const int yptr=32*offs;
  SO3part_load(offs,shared,yarr,yarrc,l2,ywidth,iix*yistride+jix*yjstride+kix*ykstride,t);

  const int rpr=32*offs+t;
  const int rpi=rpr+((r*rwidth-1)/32+1)*32;
  float* _rptr=rarr+iix*ristride+jix*rjstride+kix*rkstride+_offs;
  float* _rptri=rarrc+iix*ristride+jix*rjstride+kix*rkstride+_offs;

  if(t<rwidth){
    for(int i=0; i<r; i++)
      shared[rpr+i*rwidth]=_rptr[i*global_rwidth+t];
    for(int i=0; i<r; i++)
      shared[rpi+i*rwidth]=_rptri[i*global_rwidth+t];
  }

  __syncthreads();
  
  const int xcol=t/yfrags;
  const int xpr=xptr+xcol;
  const int xlines=((r1*xwidth-1)/32+1); 
  const int xpi=xpr+xlines*32;

  const int ycol=t%ywidth;
  const int ypr=yptr+ycol;
  const int ylines=((r2*ywidth-1)/32+1); 
  const int ypi=ypr+ylines*32;

  if(t<rwidth){
    for(int m1=-l1; m1<=l1; m1++){
      const float x_r=shared[xpr+xwidth*(m1+l1)];
      const float x_i=shared[xpi+xwidth*(m1+l1)];
      int lower=-l-m1; if(lower<-l2) lower=-l2;
      int upper=l-m1; if(upper>l2) upper=l2;
      for(int m2=lower; m2<=upper; m2++){
	float c=C_ptr[(m1+l1)*r2+m2+l2];
	const float y_r=shared[ypr+ywidth*(m2+l2)];
	const float y_i=shared[ypi+ywidth*(m2+l2)];
	shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
      }
    }
  }

  __syncthreads();

  if(t<rwidth){
    for(int i=0; i<r; i++)
      _rptr[i*global_rwidth+t]=shared[rpr+i*rwidth];
    for(int i=0; i<r; i++)
      _rptri[i*global_rwidth+t]=shared[rpi+i*rwidth];
  }

}
*/

/*
__device__ int SO3part_load(int& offs, float* shared, float* arr, float* arrc, const int l, const int w, const int skip, const int t){
  const int _offs=offs;
  int ptr=32*offs;
  const int r=2*l+1;
  const int lines=((r*w-1)/32+1); 
  float* xcell=arr+skip; 
  if(t<32)
    for(int i=0; i<lines; i++)
      shared[ptr+i*32+t]=xcell[i*32+t];
  ptr+=32*lines;
  xcell=arrc+skip;
  if(t<32)
    for(int i=0; i<lines; i++)
      shared[ptr+i*32+t]=xcell[i*32+t];
  offs+=2*lines;
  return 32*_offs;
}


__device__ int SO3part_load(int& offs, float* shared, float* arr, float* arrc, const int l, const int w, const int skip, 
  const int frag_offs, const int nfrags, const int t){
  const int _offs=offs;
  int ptr=32*offs+t;
  const int r=2*l+1;
  const int lines=((r*w-1)/32+1); 
  float* xcell=arr+skip+frag_offs+t;
  if(t<nfrags){
    for(int i=0; i<r; i++)
      shared[ptr+i*nfrags]=xcell[i*w];
    ptr+=32*lines;
    xcell=arrc+skip+frag_offs+t;
    for(int i=0; i<r; i++)
      shared[ptr+i*nfrags]=xcell[i*w];
  }
  offs+=2*lines;
  return 32*_offs;
}
*/


// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partA_DiagCGproduct_cu
#define _SO3partA_DiagCGproduct_cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/tuple.h>

//__device__ __constant__ unsigned char cg_cmem[32276]; 

#include "SO3partArrayA.hpp"
#include "SO3_CGbank.hpp"

#include "CellwiseBinaryCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "InnerCmap.hpp"
#include "OuterCmap.hpp"
#include "MVprodCmap.hpp"
#include "VMprodCmap.hpp"
#include "Convolve2Cmap.hpp"


// should move these elsewhere

__device__ void SO3part_load_lines2(float* dest, const float* source, const int nlines, const int t){
  if(t<32){
    for(int i=0; i<nlines; i++)
      dest[i*32+t]=source[i*32+t];
  }
}

__device__ void SO3part_save_lines2(const float* source, float* dest, const int nlines, const int t){
  if(t<32){
    for(int i=0; i<nlines; i++)
      dest[i*32+t]=source[i*32+t];
  }
}


// ---- DiagCGproduct --------------------------------------------------------------------------------------------


template<typename IMAP>
__global__ void SO3partA_DiagCGproduct_kernel(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const IMAP cmap, 
  const int xn, const int yn, const int rn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int r=2*l+1;
  const int r1=2*l1+1;
  const int r2=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int rwidth=xn*nch;
  const int global_rwidth=rn*nch;
  
  const int rlines=((r*rwidth-1)/32+1);
  const int xlines=((r1*xwidth-1)/32+1);
  const int ylines=((r2*ywidth-1)/32+1);

  const int rptr=0;
  const int xptr=rptr+rlines*64;
  const int yptr=xptr+xlines*64;

  int rix,xix,yix;
  int nsum;
  int lst;


  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    rix=thrust::get<0>(T);
    xix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
    //if(t==0) printf("foop1\n");
  }else{
    rix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }
  
  if(mode==1){
    if(t<32){
      for(int i=0; i<2*rlines; i++)
	shared[rptr+i*32+t]=0;
    }
  }else{
    if(t<rwidth){
      for(int i=0; i<r; i++)
	shared[rptr+i*rwidth+t]=rarr[rix*rstride+_offs+i*global_rwidth+t];
      for(int i=0; i<r; i++)
	shared[rptr+rlines*32+i*rwidth+t]=rarrc[rix*rstride+_offs+i*global_rwidth+t];
    }
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      xix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    SO3part_load_lines2(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines2(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
    SO3part_load_lines2(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines2(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);

    //if(t==0) printf("foop3\n");
      
      __syncthreads();

      const int rpr=rptr+t;
      const int rpi=rpr+rlines*32;

      const int xpr=xptr+t;
      const int xpi=xpr+xlines*32;

      const int ypr=yptr+t;
      const int ypi=ypr+ylines*32;


      if(t<rwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*r2+m2+l2];
	    const float y_r=shared[ypr+ywidth*(m2+l2)];
	    const float y_i=shared[ypi+ywidth*(m2+l2)];
	    shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	    shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
	  }
	}
      }

    //if(t==0) printf("foop4\n");

      __syncthreads();
  }

  //if(t==0) printf("fooq\n");
  
  if(t<rwidth){
    for(int i=0; i<r; i++)
      rarr[rix*rstride+_offs+i*global_rwidth+t]=shared[rptr+i*rwidth+t];
    for(int i=0; i<r; i++)
      rarrc[rix*rstride+_offs+i*global_rwidth+t]=shared[rptr+rlines*32+i*rwidth+t];
  }    
 
}



template<typename IMAP> // TODO
__global__ void SO3partA_DiagCGproduct_kernel_L(float* rarr, float* rarrc, float* xarr, float* xarrc, 
  float* yarr, float* yarrc, const int rstride, const int xstride, const int ystride, const IMAP cmap, 
  const int xn, const int yn, const int rn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int r=2*l+1;
  const int r1=2*l1+1;
  const int r2=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int rwidth=xn*nch; 
  const int global_rwidth=rn*nch;
  
  const int rlines=((r*rwidth-1)/32+1);
  const int xlines=((r1*xwidth-1)/32+1);
  const int ylines=((r2*1-1)/32+1);

  const int rptr=0;
  const int xptr=rptr+rlines*64;
  const int yptr=xptr+xlines*64;

  int rix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    rix=thrust::get<0>(T);
    xix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
  }else{
    rix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }
  

  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      xix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    SO3part_load_lines2(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines2(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);


    for(int ycol=0; ycol<ywidth; ycol++){

      if(t==0){
	for(int i=0; i<r; i++)
	  shared[yptr+i]=rarr[yix*ystride+i*ywidth+ycol];
	for(int i=0; i<r; i++)
	  shared[yptr+ylines*32+i]=rarrc[yix*ystride+i*ywidth+ycol];
      }

      if(t<32){
	for(int i=0; i<2*rlines; i++)
	  shared[rptr+i*32+t]=0;
      }
      
      __syncthreads();

      const int rpr=rptr+t;
      const int rpi=rpr+rlines*32;

      const int xcol=t;
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;

      //const int ycol=t%ywidth;
      const int ypr=yptr;// +ycol;
      const int ypi=ypr+ylines*32;


      if(t<rwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*r2+m2+l2];
	    const float y_r=shared[ypr+1*(m2+l2)];
	    const float y_i=shared[ypi+1*(m2+l2)];
	    shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	    shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
	  }
	}
      }

      //if(t==0) printf("foop4\n");

      __syncthreads();

      //if(t==0) printf("fooq\n");
  
      if(t<rwidth){
	for(int i=0; i<r; i++)
	  rarr[rix*rstride+_offs+i*global_rwidth+t*ywidth+ycol]+=shared[rptr+i*rwidth+t];
	for(int i=0; i<r; i++)
	  rarrc[rix*rstride+_offs+i*global_rwidth+t*ywidth+ycol]+=shared[rptr+rlines*32+i*rwidth+t];
      }    

      __syncthreads();

    } // ycol

  } //nsum
}




// ---- back0 ------------------------------------------------------------------------------------------------


template<typename IMAP> // TODO 
__global__ void SO3partA_DiagCGproduct_back0_kernel(float* xarr, float* xarrc, float* garr, float* garrc, 
  float* yarr, float* yarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int xptr=0;
  const int gptr=xptr+xlines*64;
  const int yptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    xix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    yix=thrust::get<2>(T);
    nsum=1;
  }else{
    xix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*xlines; i++){
	shared[xptr+i*32+t]=0;
      }
    }
  }else{
    SO3part_load_lines2(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines2(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      yix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines2(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines2(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);

    __syncthreads();

      //const int xcol=t;
    const int xpr=xptr+t;
    const int xpi=xpr+xlines*32;
    
    for(int ycol=0; ycol<ywidth; ycol++){
      
      const int ypr=yptr+ycol;
      const int ypi=ypr+ylines*32;
      
      const int gpr=gptr+ywidth*t+ycol;
      const int gpi=gpr+glines*32;

      if(t<xwidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    const float y_r=shared[ypr+ywidth*(m2+l2)];
	    const float y_i=shared[ypi+ywidth*(m2+l2)];
	    const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    shared[xpr+xwidth*(m1+l1)]+=c*(g_r*y_r+g_i*y_i); 
	    shared[xpi+xwidth*(m1+l1)]+=c*(-g_r*y_i+g_i*y_r);
	  }
	}
      }
      __syncthreads();
	
    }

  }
  
  SO3part_save_lines2(shared+xptr,xarr+xix*xstride,xlines,t);
  SO3part_save_lines2(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);
    
  __syncthreads();

}


// ---- back1 ------------------------------------------------------------------------------------------------


template<typename IMAP> // TODO 
__global__ void SO3partA_DiagCGproduct_back1_kernel(float* yarr, float* yarrc, float* garr, float* garrc, 
  float* xarr, float* xarrc, const int xstride, const int ystride, const int gstride, const IMAP cmap, 
  const int xn, const int yn, const int gn, const int l1, const int l2, const int l, 
  const int _offs, const int nch, const int Cptr, const int mode=0){

  extern __shared__ unsigned char _shared[]; 
  float* shared=reinterpret_cast<float*>(_shared);

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

  const int rg=2*l+1;
  const int rx=2*l1+1;
  const int ry=2*l2+1;

  const int xwidth=xn*nch; 
  const int ywidth=yn*nch; 
  const int gwidth=xn*yn*nch;
  const int global_gwidth=gn*nch;

  const int glines=((rg*gwidth-1)/32+1);
  const int xlines=((rx*xwidth-1)/32+1);
  const int ylines=((ry*ywidth-1)/32+1);

  const int yptr=0;
  const int gptr=yptr+ylines*64;
  const int xptr=gptr+glines*64;

  int gix,xix,yix;
  int nsum;
  int lst;

  if(mode<2){
    auto T=cmap(blockIdx.x,blockIdx.y,blockIdx.z);
    yix=thrust::get<0>(T);
    gix=thrust::get<1>(T);
    xix=thrust::get<2>(T);
    nsum=1;
  }else{
    yix=cmap.target(blockIdx.x);
    nsum=cmap.n_accum(blockIdx.x);
    lst=cmap.lst_ptr(blockIdx.x);
  }

  if(mode==1){
    if(t<32){
      for(int i=0; i<2*ylines; i++)
	shared[yptr+i*32+t]=0;
    }
  }else{
    SO3part_load_lines2(shared+yptr,yarr+yix*ystride,ylines,t);
    SO3part_load_lines2(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
  }
  
  for(int s=0; s<nsum; s++){

    if(mode==2){
      auto T=cmap.source(lst,blockIdx.x,s);
      gix=thrust::get<0>(T);
      xix=thrust::get<1>(T);
    }

    // hack: gwidth assumed to be <=32
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+i*gwidth+t]=garr[gix*gstride+i*global_gwidth+_offs+t];
    for(int i=0; i<rg; i++)
      if(t<gwidth)
	shared[gptr+glines*32+i*gwidth+t]=garrc[gix*gstride+i*global_gwidth+_offs+t];

    SO3part_load_lines2(shared+xptr,xarr+xix*xstride,xlines,t);
    SO3part_load_lines2(shared+xptr+xlines*32,xarrc+xix*xstride,xlines,t);

    __syncthreads();

    //const int ycol=t;
    const int ypr=yptr+t;
    const int ypi=ypr+ylines*32;
    
    for(int xcol=0; xcol<xwidth; xcol++){
      
      const int xpr=xptr+xcol;
      const int xpi=xpr+xlines*32;
      
      const int gpr=gptr+ywidth*xcol+t;
      const int gpi=gpr+glines*32;

      if(t<ywidth){
	for(int m1=-l1; m1<=l1; m1++){
	  int lower=-l-m1; if(lower<-l2) lower=-l2;
	  int upper=l-m1; if(upper>l2) upper=l2;
	  const float x_r=shared[xpr+xwidth*(m1+l1)];
	  const float x_i=shared[xpi+xwidth*(m1+l1)];
	  for(int m2=lower; m2<=upper; m2++){
	    float c=C_ptr[(m1+l1)*ry+m2+l2];
	    const float g_r=shared[gpr+gwidth*(m1+m2+l)];
	    const float g_i=shared[gpi+gwidth*(m1+m2+l)];
	    shared[ypr+ywidth*(m2+l2)]+=c*(g_r*x_r+g_i*x_i); 
	    shared[ypi+ywidth*(m2+l2)]+=c*(-g_r*x_i+g_i*x_r);
	  }
	}
      }
      __syncthreads();

    }

  }
  
  SO3part_save_lines2(shared+yptr,yarr+yix*ystride,ylines,t);
  SO3part_save_lines2(shared+yptr+ylines*32,yarrc+yix*ystride,ylines,t);
    
  __syncthreads();

}


// -----------------------------------------------------------------------------------------------------------


namespace GElib{


  template<typename CMAP>
  void SO3partA_DiagCGproduct_cu(const CMAP& map, SO3partArrayA& r, const SO3partArrayA& x, 
    const SO3partArrayA& y, const cudaStream_t& stream, const int offs, const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=r.getl();
    const int _nch=1;
    assert(x.nbu==r.nbu);
    assert(y.nbu==r.nbu);
    int _nbu=1; if(_nbu<0) _nbu=1;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    //int nlines=x.cellstride/16+y.cellstride/16+r.cellstride/16; // should be smaller than this!
    int nlines=x.cellstride/16+y.cellstride/16+cnine::roundup(x.getn()*_nch*(2*l+1),32)/16;
    // nlines/=_nbu;

    if(nlines<=384){

      SO3partA_DiagCGproduct_kernel<<<map.blockdims(),cnine::roundup(x.getn(),32),nlines*128,stream>>>
	(r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	  r.cellstride,x.cellstride,y.cellstride,map,
	  x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);

    }else{ // TODO 
      
      int nlines=x.cellstride/16+cnine::roundup(_nch*(2*yl+1),32)/16+cnine::roundup(x.getn()*_nch*(2*l+1),32)/16;

      if(nlines>384){
	cout<<"GElib error: DiagCGproduct too big for shared memory"<<endl;
      }else{
	SO3partA_DiagCGproduct_kernel_L<<<map.blockdims(),cnine::roundup(x.getn(),32),nlines*128,stream>>>
	  (r.arrg,r.arrgc,x.arrg,x.arrgc,y.arrg,y.arrgc,
	    r.cellstride,x.cellstride,y.cellstride,map,
	    x.getn(),y.getn(),r.getn(),xl,yl,l,offs,_nch,Cptr,mode);
      }
    }

  }

  
  template<typename CMAP> // TODO 
  void SO3partA_DiagCGproduct_back0_cu(const CMAP& map, SO3partArrayA& x, const SO3partArrayA& g, 
    const SO3partArrayA& y, const cudaStream_t& stream, const int offs, const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=g.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+g.cellstride/16;
    assert(x.nbu==g.nbu);
    assert(y.nbu==g.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    nlines/=_nbu;

    SO3partA_DiagCGproduct_back0_kernel<<<map.blockdims(),cnine::roundup(x.getn(),32),nlines*128,stream>>>
      (x.arrg,x.arrgc,g.arrg,g.arrgc,y.arrg,y.arrgc,
	x.cellstride,y.cellstride,g.cellstride,map,
	x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);

  }

  
  template<typename CMAP> // TODO 
  void SO3partA_DiagCGproduct_back1_cu(const CMAP& map, SO3partArrayA& y, const SO3partArrayA& g, 
    const SO3partArrayA& x, const cudaStream_t& stream, const int offs, const int mode){

    const int xl=x.getl();
    const int yl=y.getl();
    const int l=g.getl();

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
    int nlines=x.cellstride/16+y.cellstride/16+g.cellstride/16;
    assert(x.nbu==g.nbu);
    assert(y.nbu==g.nbu);

    const int _nch=1;
    int _nbu=1; if(_nbu<0) _nbu=1;
    nlines/=_nbu;

    SO3partA_DiagCGproduct_back1_kernel<<<map.blockdims(),cnine::roundup(y.getn(),32),nlines*128,stream>>>
      (y.arrg,y.arrgc,g.arrg,g.arrgc,x.arrg,x.arrgc,
	x.cellstride,y.cellstride,g.cellstride,map,
	x.getn(),y.getn(),g.getn(),xl,yl,l,offs,_nch,Cptr,mode);

  }


  template void SO3partA_DiagCGproduct_cu(const cnine::CellwiseBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_DiagCGproduct_cu(const cnine::BroadcastBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_DiagCGproduct_cu(const cnine::OuterCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_DiagCGproduct_cu(const cnine::InnerCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_DiagCGproduct_cu(const cnine::MVprodCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_DiagCGproduct_cu(const cnine::Convolve2Cmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);



  template void SO3partA_DiagCGproduct_back0_cu(const cnine::CellwiseBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_DiagCGproduct_back0_cu(const cnine::BroadcastBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_DiagCGproduct_back0_cu(const cnine::OuterCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);


  template void SO3partA_DiagCGproduct_back1_cu(const cnine::CellwiseBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_DiagCGproduct_back1_cu(const cnine::BroadcastBinaryCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);

  template void SO3partA_DiagCGproduct_back1_cu(const cnine::OuterCmap& map, 
    SO3partArrayA&, const SO3partArrayA&, const SO3partArrayA&, const cudaStream_t&, const int offs, 
    const int mode);




}

#endif 





// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022s, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partB_addCGproduct_back0_cu
#define _SO3partB_addCGproduct_back0_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"

//__device__ __constant__ unsigned char cg_cmem[32276]; 

extern GElib::SO3_CGbank SO3_cgbank;


__device__ int loadg1(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=sourcec[i*s1+t*s2];
  }
  return offs;
}


__device__ int saveg1(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=sourcec[i*J+t];
  }
  return offs;
}


__global__ void SO3partB_addCGproduct_back0_kernel(const cnine::Ctensor3_view x, const cnine::Ctensor3_view r, 
  const cnine::Ctensor3_view y, const int Cptr){

  extern __shared__ unsigned char _shared[]; 
  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int b=blockIdx.x;
  const int t=threadIdx.x;

//printf("%d",t);

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int xn=x.n2;
  int yn=y.n2;
  int rn=xn*yn;
  int L2=y.n1;

//for(int i=0; i<x.n1; i++)
//for(int j=0; j<y.n1; j++)
//if(t==0)printf("%f\n",C_ptr[i*L2+j]);

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg1(x,xpr,b,t);

  float* ypr=xpr+((2*x.n1*xn-1)/32+1)*32;
  float* ypi=ypr+loadg1(y,ypr,b,t);

  float* rpr=ypr+((2*y.n1*yn-1)/32+1)*32;
  float* rpi=rpr+loadg1(r,rpr,b,t);

  __syncthreads();


  float* _xpr=xpr+t;
  float* _xpi=xpi+t;

  for(int ycol=0; ycol<yn; ycol++){
    if(t<xn){

      float* _ypr=ypr+ycol;
      float* _ypi=ypi+ycol;
      
      float* _rpr=rpr+t*yn+ycol;
      float* _rpi=rpi+t*yn+ycol;
      
      for(int m1=-l1; m1<=l1; m1++){
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=C_ptr[(m1+l1)*L2+m2+l2];
	  const float y_r=_ypr[yn*(m2+l2)];
	  const float y_i=_ypi[yn*(m2+l2)];
	  const float g_r=_rpr[rn*(m1+m2+l)];
	  const float g_i=_rpi[rn*(m1+m2+l)];
	  _xpr[xn*(m1+l1)]+=c*(g_r*y_r+g_i*y_i);
	  _xpi[xn*(m1+l1)]+=c*(-g_r*y_i+g_i*y_r);
	}
      }
    }
    __syncthreads();
  }
  

  __syncthreads();
  
  saveg1(x,xpr,b,t);

}



namespace GElib{


  void SO3partB_addCGproduct_back0_cu(const cnine::Ctensor3_view& xg, cnine::Ctensor3_view rg, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream){

    const int xl=(xg.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(rg.n1-1)/2;

    const int b=rg.n0;
    assert(xg.n0==b);
    assert(y.n0==b);

    rg.arr+=rg.s2*offs;
    rg.arrc+=rg.s2*offs;
    rg.n2=xg.n2*y.n2;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;

    int nlines=cnine::roundup(xg.n1*xg.n2*2,32)/32+
      cnine::roundup(y.n1*y.n2*2,32)/32+
      cnine::roundup(rg.n1*xg.n2*y.n2*2,32)/32;

    if(nlines<=384){

      SO3partB_addCGproduct_back0_kernel<<<b,cnine::roundup(xg.n2*y.n2,32),nlines*128,stream>>>
	(xg,rg,y,Cptr);

    }else{
      cout<<"error"<<endl;
    }

    //r.arr-=r.s1*offs;
    //r.arrc-=r.s1*offs;
    //r.n1=rn1;

  }    


}


#endif 



  /*
  if(t<32){
    int xn=xview.n1;
    int xs0=xview.s0;
    int xs1=xview.s1;
    int xarr=xview.arr;
    int xarrc=xview.arrc;
    for(int i=0; i<2*l1+1; i++)
      for(int j=0; j<xn; x++)
	xpr[i*xwidth+j]=xarr[i*xs0+j*xs1];
    for(int i=0; i<2*l1+1; i++)
      for(int j=0; j<xn; x++)
	xpi[i*xwidth+j]=xarrc[i*xs0+j*xs1];
  }

  if(t<32){
    int yn=yview.n1;
    int ys0=yview.s0;
    int ys1=yview.s1;
    int yarr=yview.arr;
    int yarrc=yview.arrc;
    for(int i=0; i<2*l2+1; i++)
      for(int j=0; j<xn; x++)
	ypr[i*ywidth+j]=yarr[i*ys0+j*ys1];
    for(int i=0; i<2*l2+1; i++)
      for(int j=0; j<xn; x++)
	ypi[i*ywidth+j]=yarrc[i*ys0+j*ys1];
  }

  if(t<rwidth){
    for(int m1=-l1; m1<=l1; m1++){
      const float x_r=xpr[xwidth*(m1+l1)];
      const float x_i=xpi[xwidth*(m1+l1)];
      int lower=-l-m1; if(lower<-l2) lower=-l2;
      int upper=l-m1; if(upper>l2) upper=l2;
      for(int m2=lower; m2<=upper; m2++){
	float c=C_ptr[(m1+l1)*r2+m2+l2];
	const float y_r=shared[ypr+ywidth*(m2+l2)];
	const float y_i=shared[ypi+ywidth*(m2+l2)];
	shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
      }
    }
  }
  */
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022s, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partB_addCGproduct_back1_cu
#define _SO3partB_addCGproduct_back1_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor3_view.hpp"

//__device__ __constant__ unsigned char cg_cmem[32276]; 

extern GElib::SO3_CGbank SO3_cgbank;


__device__ int loadg2(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=sourcec[i*s1+t*s2];
  }
  return offs;
}


__device__ int saveg2(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=sourcec[i*J+t];
  }
  return offs;
}


__global__ void SO3partB_addCGproduct_back1_kernel(const cnine::Ctensor3_view y, const cnine::Ctensor3_view r, 
  const cnine::Ctensor3_view x, const int Cptr){

  extern __shared__ unsigned char _shared[]; 
  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int b=blockIdx.x;
  const int t=threadIdx.x;

//printf("%d",t);

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int xn=x.n2;
  int yn=y.n2;
  int rn=xn*yn;
  int L2=y.n1;

//for(int i=0; i<x.n1; i++)
//for(int j=0; j<y.n1; j++)
//if(t==0)printf("%f\n",C_ptr[i*L2+j]);

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg2(x,xpr,b,t);

  float* ypr=xpr+((2*x.n1*xn-1)/32+1)*32;
  float* ypi=ypr+loadg2(y,ypr,b,t);

  float* rpr=ypr+((2*y.n1*yn-1)/32+1)*32;
  float* rpi=rpr+loadg2(r,rpr,b,t);

  __syncthreads();



  for(int xcol=0; xcol<xn; xcol++){
    if(t<yn){

      float* _xpr=xpr+xcol;
      float* _xpi=xpi+xcol;

      float* _ypr=ypr+t;
      float* _ypi=ypi+t;
      
      float* _rpr=rpr+xcol*yn+t;
      float* _rpi=rpi+xcol*yn+t;
      
      for(int m1=-l1; m1<=l1; m1++){
	const float x_r=_xpr[xn*(m1+l1)];
	const float x_i=_xpi[xn*(m1+l1)];
	int lower=-l-m1; if(lower<-l2) lower=-l2;
	int upper=l-m1; if(upper>l2) upper=l2;
	for(int m2=lower; m2<=upper; m2++){
	  float c=C_ptr[(m1+l1)*L2+m2+l2];
	  const float g_r=_rpr[rn*(m1+m2+l)];
	  const float g_i=_rpi[rn*(m1+m2+l)];
	  _ypr[yn*(m2+l2)]+=c*(g_r*x_r+g_i*x_i);
	  _ypi[yn*(m2+l2)]+=c*(-g_r*x_i+g_i*x_r);
	}
      }
    }
    __syncthreads();
  }
  

  __syncthreads();
  
  saveg2(y,ypr,b,t);

}



namespace GElib{


  void SO3partB_addCGproduct_back1_cu(const cnine::Ctensor3_view& yg, cnine::Ctensor3_view g, const cnine::Ctensor3_view& x, 
    const int offs, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(yg.n1-1)/2;
    const int l=(g.n1-1)/2;

    const int b=g.n0;
    assert(x.n0==b);
    assert(yg.n0==b);

    g.arr+=g.s2*offs;
    g.arrc+=g.s2*offs;
    g.n2=x.n2*yg.n2;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;

    int nlines=cnine::roundup(x.n1*x.n2*2,32)/32+
      cnine::roundup(yg.n1*yg.n2*2,32)/32+
      cnine::roundup(g.n1*x.n2*yg.n2*2,32)/32;

    if(nlines<=384){

      SO3partB_addCGproduct_back1_kernel<<<b,cnine::roundup(x.n2*yg.n2,32),nlines*128,stream>>>
	(yg,g,x,Cptr);

    }else{
      cout<<"error"<<endl;
    }

    //r.arr-=r.s1*offs;
    //r.arrc-=r.s1*offs;
    //r.n1=rn1;

  }    


}


#endif 


// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022s, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partB_addCGproduct_cu
#define _SO3partB_addCGproduct_cu

#include <cuda.h>
#include <cuda_runtime.h>
//#include <thrust/complex.h>
//#include <thrust/tuple.h>

#include "SO3_CGbank.hpp"
#include "Ctensor2_view.hpp"
#include "Ctensor3_view.hpp"

//__device__ __constant__ unsigned char cg_cmem[32276]; 

extern GElib::SO3_CGbank SO3_cgbank;


__device__ int loadg(const cnine::Ctensor2_view& x, float* dest, const int t){
  int I=x.n0;
  int J=x.n1;
  int s0=x.s0;
  int s1=x.s1;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=x.arr[i*s0+t*s1];
    for(int i=0; i<I; i++)
      destc[i*J+t]=x.arrc[i*s0+t*s1];
  }
  return offs;
}


__device__ int saveg(const cnine::Ctensor2_view& x, float* source, const int t){
  int I=x.n0;
  int J=x.n1;
  int s0=x.s0;
  int s1=x.s1;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  if(t<J){
    for(int i=0; i<I; i++)
      x.arr[i*s0+t*s1]=source[i*J+t];
    for(int i=0; i<I; i++)
      x.arrc[i*s0+t*s1]=sourcec[i*J+t];
  }
  return offs;
}


__device__ int loadg(const cnine::Ctensor3_view& x, float* dest, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* destc=dest+offs;
  float* source=x.arr+x.s0*b;
  float* sourcec=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*J+t]=source[i*s1+t*s2];
    for(int i=0; i<I; i++)
      destc[i*J+t]=sourcec[i*s1+t*s2];
  }
  return offs;
}


__device__ int saveg(const cnine::Ctensor3_view& x, float* source, const int b, const int t){
  int I=x.n1;
  int J=x.n2;
  int s1=x.s1;
  int s2=x.s2;
  int offs=I*J; //((I*J-1)/32+1)*32;
  float* sourcec=source+offs;
  float* dest=x.arr+x.s0*b;
  float* destc=x.arrc+x.s0*b;
  if(t<J){
    for(int i=0; i<I; i++)
      dest[i*s1+t*s2]=source[i*J+t];
    for(int i=0; i<I; i++)
      destc[i*s1+t*s2]=sourcec[i*J+t];
  }
  return offs;
}




__global__ void SO3partB_addCGproduct_kernel(const cnine::Ctensor2_view r, const cnine::Ctensor2_view x, 
  const cnine::Ctensor2_view y, const int Cptr){


  extern __shared__ unsigned char _shared[]; 

  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int t=threadIdx.x;

//printf("%d",t);

  int l1=(x.n0-1)/2;
  int l2=(y.n0-1)/2;
  int l=(r.n0-1)/2;
  int xn=x.n1;
  int yn=y.n1;
  int rn=xn*yn;
  int L2=y.n0;

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg(x,xpr,t);

  float* ypr=xpr+((2*x.n0*xn-1)/32+1)*32;
  float* ypi=ypr+loadg(y,ypr,t);

  float* rpr=ypr+((2*y.n0*yn-1)/32+1)*32;
  float* rpi=rpr+loadg(r,rpr,t);

  __syncthreads();

  if(t<rn){

    xpr=xpr+t/yn;
    xpi=xpi+t/yn;
    
    ypr=ypr+t%yn;
    ypi=ypi+t%yn;
    
    float* _rpr=rpr+t;
    float* _rpi=rpi+t;

    for(int m1=-l1; m1<=l1; m1++){
      const float x_r=xpr[xn*(m1+l1)];
      const float x_i=xpi[xn*(m1+l1)];
      int lower=-l-m1; if(lower<-l2) lower=-l2;
      int upper=l-m1; if(upper>l2) upper=l2;
      for(int m2=lower; m2<=upper; m2++){
	float c=C_ptr[(m1+l1)*L2+m2+l2];
	const float y_r=ypr[yn*(m2+l2)];
	const float y_i=ypi[yn*(m2+l2)];
	_rpr[rn*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	_rpi[rn*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
      }
    }
  }

  __syncthreads();
  
  saveg(r,rpr,t);

}

__global__ void SO3partB_addCGproduct_kernel(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, 
  const cnine::Ctensor3_view y, const int Cptr){

  extern __shared__ unsigned char _shared[]; 
  const float* C_ptr=reinterpret_cast<float*>(cg_cmem)+Cptr;
  const int b=blockIdx.x;
  const int t=threadIdx.x;

//printf("%d",t);

  int l1=(x.n1-1)/2;
  int l2=(y.n1-1)/2;
  int l=(r.n1-1)/2;
  int xn=x.n2;
  int yn=y.n2;
  int rn=xn*yn;
  int L2=y.n1;

  float* xpr=reinterpret_cast<float*>(_shared);
  float* xpi=xpr+loadg(x,xpr,b,t);

  float* ypr=xpr+((2*x.n1*xn-1)/32+1)*32;
  float* ypi=ypr+loadg(y,ypr,b,t);

  float* rpr=ypr+((2*y.n1*yn-1)/32+1)*32;
  float* rpi=rpr+loadg(r,rpr,b,t);

  __syncthreads();

  if(t<rn){

    xpr=xpr+t/yn;
    xpi=xpi+t/yn;
    
    ypr=ypr+t%yn;
    ypi=ypi+t%yn;
    
    float* _rpr=rpr+t;
    float* _rpi=rpi+t;

    for(int m1=-l1; m1<=l1; m1++){
      const float x_r=xpr[xn*(m1+l1)];
      const float x_i=xpi[xn*(m1+l1)];
      int lower=-l-m1; if(lower<-l2) lower=-l2;
      int upper=l-m1; if(upper>l2) upper=l2;
      for(int m2=lower; m2<=upper; m2++){
	float c=C_ptr[(m1+l1)*L2+m2+l2];
	const float y_r=ypr[yn*(m2+l2)];
	const float y_i=ypi[yn*(m2+l2)];
	_rpr[rn*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	_rpi[rn*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
      }
    }
  }

  __syncthreads();
  
  saveg(r,rpr,b,t);

}



namespace GElib{

  void SO3partB_addCGproduct_cu(cnine::Ctensor2_view r, const cnine::Ctensor2_view& x, const cnine::Ctensor2_view& y, 
    const cudaStream_t& stream, const int offs=0){

    const int xl=(x.n0-1)/2;
    const int yl=(y.n0-1)/2;
    const int l=(r.n0-1)/2;
    r.arr+=r.s1*offs;
    r.arrc+=r.s1*offs;
    //int rn1=r.n1;
    r.n1=x.n1*y.n1;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;

    int nlines=cnine::roundup(x.n0*x.n1*2,32)/32+
      cnine::roundup(y.n0*y.n1*2,32)/32+
      cnine::roundup(r.n0*x.n1*y.n1*2,32)/32;


    if(nlines<=384){

      SO3partB_addCGproduct_kernel<<<1,cnine::roundup(x.n1*y.n1,32),nlines*128,stream>>>
	(r,x,y,Cptr);

    }else{
      cout<<"error"<<endl;
    }

    //r.arr-=r.s1*offs;
    //r.arrc-=r.s1*offs;
    //r.n1=rn1;

  }    


  void SO3partB_addCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream){

    const int xl=(x.n1-1)/2;
    const int yl=(y.n1-1)/2;
    const int l=(r.n1-1)/2;

    const int b=r.n0;
    assert(x.n0==b);
    assert(y.n0==b);

    r.arr+=r.s2*offs;
    r.arrc+=r.s2*offs;
    r.n2=x.n2*y.n2;

    int Cptr=SO3_cgbank.getfC(xl,yl,l)/4;
//cout<<"Cptr="<<Cptr<<endl;


    int nlines=cnine::roundup(x.n1*x.n2*2,32)/32+
      cnine::roundup(y.n1*y.n2*2,32)/32+
      cnine::roundup(r.n1*x.n2*y.n2*2,32)/32;


    if(nlines<=384){

      SO3partB_addCGproduct_kernel<<<b,cnine::roundup(x.n2*y.n2,32),nlines*128,stream>>>
	(r,x,y,Cptr);

    }else{
      cout<<"error"<<endl;
    }

    //r.arr-=r.s1*offs;
    //r.arrc-=r.s1*offs;
    //r.n1=rn1;

  }    


}


#endif 



  /*
  if(t<32){
    int xn=xview.n1;
    int xs0=xview.s0;
    int xs1=xview.s1;
    int xarr=xview.arr;
    int xarrc=xview.arrc;
    for(int i=0; i<2*l1+1; i++)
      for(int j=0; j<xn; x++)
	xpr[i*xwidth+j]=xarr[i*xs0+j*xs1];
    for(int i=0; i<2*l1+1; i++)
      for(int j=0; j<xn; x++)
	xpi[i*xwidth+j]=xarrc[i*xs0+j*xs1];
  }

  if(t<32){
    int yn=yview.n1;
    int ys0=yview.s0;
    int ys1=yview.s1;
    int yarr=yview.arr;
    int yarrc=yview.arrc;
    for(int i=0; i<2*l2+1; i++)
      for(int j=0; j<xn; x++)
	ypr[i*ywidth+j]=yarr[i*ys0+j*ys1];
    for(int i=0; i<2*l2+1; i++)
      for(int j=0; j<xn; x++)
	ypi[i*ywidth+j]=yarrc[i*ys0+j*ys1];
  }

  if(t<rwidth){
    for(int m1=-l1; m1<=l1; m1++){
      const float x_r=xpr[xwidth*(m1+l1)];
      const float x_i=xpi[xwidth*(m1+l1)];
      int lower=-l-m1; if(lower<-l2) lower=-l2;
      int upper=l-m1; if(upper>l2) upper=l2;
      for(int m2=lower; m2<=upper; m2++){
	float c=C_ptr[(m1+l1)*r2+m2+l2];
	const float y_r=shared[ypr+ywidth*(m2+l2)];
	const float y_i=shared[ypi+ywidth*(m2+l2)];
	shared[rpr+rwidth*(m1+m2+l)]+=c*(x_r*y_r-x_i*y_i); 
	shared[rpi+rwidth*(m1+m2+l)]+=c*(x_r*y_i+x_i*y_r);
      }
    }
  }
  */
