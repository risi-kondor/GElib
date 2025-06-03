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


#ifndef _CnineCtensorEinsumFn
#define _CnineCtensorEinsumFn

#include "EinsumFnBase.hpp"
#include "CtensorView.hpp"


namespace cnine{


  template<typename TYPE>
  class CtensorEinsumFn: public EinsumFnBase{
  public:


    CtensorEinsumFn(const string str): 
      EinsumFnBase(str){
    }

      
  public:

    void operator()(const CtensorView& r, const CtensorView& x, const CtensorView& y){

      if(dstrides.size()==0){
	sloops(r,x,y,0,0,0);
	return;
      }

      if(dstrides.size()==1){

	int I0=r.dims[dstrides[0].first[0]];
	int r0=r.strides.combine(dstrides[0].first);
	int x0=x.strides.combine(dstrides[0].second);
	int y0=y.strides.combine(dstrides[0].third);

	for(int i0=0; i0<I0; i0++)
	  sloops(r,x,y,i0*r0,i0*x0,i0*y0);
	return;
      }

      if(dstrides.size()==2){

	int I0=r.dims[dstrides[0].first[0]];
	int r0=r.strides.combine(dstrides[0].first);
	int x0=x.strides.combine(dstrides[0].second);
	int y0=y.strides.combine(dstrides[0].third);

	int I1=r.dims[dstrides[1].first[0]];
	int r1=r.strides.combine(dstrides[1].first);
	int x1=x.strides.combine(dstrides[1].second);
	int y1=y.strides.combine(dstrides[1].third);

	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    sloops(r,x,y,i0*r0+i1*r1,i0*x0+i1*x1,i0*y0+i1*y1);

	return;
      }

      if(dstrides.size()==3){

	int I0=r.dims[dstrides[0].first[0]];
	int r0=r.strides.combine(dstrides[0].first);
	int x0=x.strides.combine(dstrides[0].second);
	int y0=y.strides.combine(dstrides[0].third);

	int I1=r.dims[dstrides[1].first[0]];
	int r1=r.strides.combine(dstrides[1].first);
	int x1=x.strides.combine(dstrides[1].second);
	int y1=y.strides.combine(dstrides[1].third);

	int I2=r.dims[dstrides[2].first[0]];
	int r2=r.strides.combine(dstrides[2].first);
	int x2=x.strides.combine(dstrides[2].second);
	int y2=y.strides.combine(dstrides[2].third);

	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    for(int i2=0; i2<I2; i2++)
	    sloops(r,x,y,i0*r0+i1*r1+i2*r2,i0*x0+i1*x1+i2*x2,i0*y0+i1*y1+i2*y2);

	return;
      }
      
      if(dstrides.size()==4){

	int I0=r.dims[dstrides[0].first[0]];
	int r0=r.strides.combine(dstrides[0].first);
	int x0=x.strides.combine(dstrides[0].second);
	int y0=y.strides.combine(dstrides[0].third);

	int I1=r.dims[dstrides[1].first[0]];
	int r1=r.strides.combine(dstrides[1].first);
	int x1=x.strides.combine(dstrides[1].second);
	int y1=y.strides.combine(dstrides[1].third);

	int I2=r.dims[dstrides[2].first[0]];
	int r2=r.strides.combine(dstrides[2].first);
	int x2=x.strides.combine(dstrides[2].second);
	int y2=y.strides.combine(dstrides[2].third);

	int I3=r.dims[dstrides[3].first[0]];
	int r3=r.strides.combine(dstrides[3].first);
	int x3=x.strides.combine(dstrides[3].second);
	int y3=y.strides.combine(dstrides[3].third);

	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    for(int i2=0; i2<I2; i2++)
	      for(int i3=0; i3<I3; i3++){
		sloops(r,x,y,i0*r0+i1*r1+i2*r2+i3*r3,i0*x0+i1*x1+i2*x2+i3*x3,i0*y0+i1*y1+i2*y2+i3*y3);
	      }

	return;
      }
      
    }


  private:


    inline void sloops(const CtensorView& r, const CtensorView& x, const CtensorView& y, int ro, int xo, int yo){
      
      float xfact=1; if(xconj) xfact=-1;
      int yfact=1; if(xconj) xfact=-1;

      if(sstrides.size()==0){
	if(xconj){
	  if(yconj){
	    bloops(r,ro,complex<TYPE>(x.arr[xo],-x.arrc[xo])*
	      complex<TYPE>(y.arr[yo],-y.arrc[yo]));
	  }else{
	    bloops(r,ro,complex<TYPE>(x.arr[xo],-x.arrc[xo])*
	      complex<TYPE>(y.arr[yo],y.arrc[yo]));
	  }
	}else{
	  if(yconj)
	    bloops(r,ro,complex<TYPE>(x.arr[xo],x.arrc[xo])*
	      complex<TYPE>(y.arr[yo],-y.arrc[yo]));
	  else
	    bloops(r,ro,complex<TYPE>(x.arr[xo],x.arrc[xo])*
	      complex<TYPE>(y.arr[yo],y.arrc[yo]));
	}
	return;
      }

      if(sstrides.size()==1){
	int I0=x.dims[sstrides[0].first[0]];
	int x0=x.strides.combine(sstrides[0].first);
	int y0=y.strides.combine(sstrides[0].second);

	if(xconj){
	  if(yconj){
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      t+=complex<TYPE>(x.arr[xo+i0*x0],-x.arrc[xo+i0*x0])*
		complex<TYPE>(y.arr[yo+i0*y0],-y.arrc[yo+i0*y0]);
	    bloops(r,ro,t);
	  }else{
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      t+=complex<TYPE>(x.arr[xo+i0*x0],-x.arrc[xo+i0*x0])*
		complex<TYPE>(y.arr[yo+i0*y0],y.arrc[yo+i0*y0]);
	    bloops(r,ro,t);
	  }
	}else{
	  if(yconj){
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      t+=complex<TYPE>(x.arr[xo+i0*x0],x.arrc[xo+i0*x0])*
		complex<TYPE>(y.arr[yo+i0*y0],-y.arrc[yo+i0*y0]);
	    bloops(r,ro,t);
	  }else{
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++){
	      t+=complex<TYPE>(x.arr[xo+i0*x0],x.arrc[xo+i0*x0])*
		complex<TYPE>(y.arr[yo+i0*y0],y.arrc[yo+i0*y0]);
	    }
	    bloops(r,ro,t);
	  }
	}
	return;
      }


      if(sstrides.size()==2){
	int I0=x.dims[sstrides[0].first[0]];
	int x0=x.strides.combine(sstrides[0].first);
	int y0=y.strides.combine(sstrides[0].second);

	int I1=x.dims[sstrides[1].first[0]];
	int x1=x.strides.combine(sstrides[1].first);
	int y1=y.strides.combine(sstrides[1].second);

	if(xconj){
	  if(yconj){
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      for(int i1=0; i1<I1; i1++)
		t+=complex<TYPE>(x.arr[xo+i0*x0+i1*x1],-x.arrc[xo+i0*x0+i1*x1])*
		  complex<TYPE>(y.arr[yo+i0*y0+i1*y1],-y.arrc[yo+i0*y0+i1*y1]);
	    bloops(r,ro,t);
	  }else{
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      for(int i1=0; i1<I1; i1++)
		t+=complex<TYPE>(x.arr[xo+i0*x0+i1*x1],-x.arrc[xo+i0*x0+i1*x1])*
		  complex<TYPE>(y.arr[yo+i0*y0+i1*y1],y.arrc[yo+i0*y0+i1*y1]);
	    bloops(r,ro,t);
	  }
	}else{
	  if(yconj){
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      for(int i1=0; i1<I1; i1++)
		t+=complex<TYPE>(x.arr[xo+i0*x0+i1*x1],x.arrc[xo+i0*x0+i1*x1])*
		  complex<TYPE>(y.arr[yo+i0*y0+i1*y1],-y.arrc[yo+i0*y0+i1*y1]);
	    bloops(r,ro,t);
	  }else{
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      for(int i1=0; i1<I1; i1++)
		t+=complex<TYPE>(x.arr[xo+i0*x0+i1*x1],x.arrc[xo+i0*x0+i1*x1])*
		  complex<TYPE>(y.arr[yo+i0*y0+i1*y1],y.arrc[yo+i0*y0+i1*y1]);
	    bloops(r,ro,t);
	  }
	}
	return;
      }

      if(sstrides.size()==3){
	int I0=x.dims[sstrides[0].first[0]];
	int x0=x.strides.combine(sstrides[0].first);
	int y0=y.strides.combine(sstrides[0].second);

	int I1=x.dims[sstrides[1].first[0]];
	int x1=x.strides.combine(sstrides[1].first);
	int y1=y.strides.combine(sstrides[1].second);

	int I2=x.dims[sstrides[2].first[0]];
	int x2=x.strides.combine(sstrides[2].first);
	int y2=y.strides.combine(sstrides[2].second);

	if(xconj){
	  if(yconj){
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      for(int i1=0; i1<I1; i1++)
		for(int i2=0; i2<I2; i2++)
		  t+=complex<TYPE>(x.arr[xo+i0*x0+i1*x1+i2*x2],-x.arrc[xo+i0*x0+i1*x1+i2*x2])*
		    complex<TYPE>(y.arr[yo+i0*y0+i1*y1+i2*y2],y.arrc[yo+i0*y0+i1*y1+i2*y2]);
	    bloops(r,ro,t);
	  }else{
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      for(int i1=0; i1<I1; i1++)
		for(int i2=0; i2<I2; i2++)
		  t+=complex<TYPE>(x.arr[xo+i0*x0+i1*x1+i2*x2],-x.arrc[xo+i0*x0+i1*x1+i2*x2])*
		    complex<TYPE>(y.arr[yo+i0*y0+i1*y1+i2*y2],-y.arrc[yo+i0*y0+i1*y1+i2*y2]);
	    bloops(r,ro,t);
	  }
	}else{
	  if(yconj){
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      for(int i1=0; i1<I1; i1++)
		for(int i2=0; i2<I2; i2++)
		  t+=complex<TYPE>(x.arr[xo+i0*x0+i1*x1+i2*x2],x.arrc[xo+i0*x0+i1*x1+i2*x2])*
		    complex<TYPE>(y.arr[yo+i0*y0+i1*y1+i2*y2],-y.arrc[yo+i0*y0+i1*y1+i2*y2]);
	    bloops(r,ro,t);
	  }else{
	    complex<TYPE> t=0;
	    for(int i0=0; i0<I0; i0++)
	      for(int i1=0; i1<I1; i1++)
		for(int i2=0; i2<I2; i2++)
		  t+=complex<TYPE>(x.arr[xo+i0*x0+i1*x1+i2*x2],x.arrc[xo+i0*x0+i1*x1+i2*x2])*
		    complex<TYPE>(y.arr[yo+i0*y0+i1*y1+i2*y2],y.arrc[yo+i0*y0+i1*y1+i2*y2]);
	    bloops(r,ro,t);
	  }
	}
	return;

      }
    }
      


    inline void bloops(const CtensorView& r, int roffs, complex<TYPE> t){

      if(bstrides.size()==0){
	r.arr[roffs]+=std::real(t);
	r.arrc[roffs]+=std::imag(t);
	return;
      }


      if(bstrides.size()==1){
	int I0=r.dims[bstrides[0][0]];
	int r0=r.strides.combine(bstrides[0]);

	for(int i0=0; i0<I0; i0++){
	  r.arr[roffs+i0*r0]+=std::real(t);
	  r.arrc[roffs+i0*r0]+=std::imag(t);
	}
	return;
      }


      if(bstrides.size()==2){
	int I0=r.dims[bstrides[0][0]];
	int r0=r.strides.combine(bstrides[0]);
	int I1=r.dims[bstrides[1][0]];
	int r1=r.strides.combine(bstrides[1]);

	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++){
	    r.arr[roffs+i0*r0+i1*r1]+=std::real(t);
	    r.arrc[roffs+i0*r0+i1*r1]+=std::imag(t);
	  }
	return;
      }


      if(bstrides.size()==3){
	int I0=r.dims[bstrides[0][0]];
	int r0=r.strides.combine(bstrides[0]);
	int I1=r.dims[bstrides[1][0]];
	int r1=r.strides.combine(bstrides[1]);
	int I2=r.dims[bstrides[2][0]];
	int r2=r.strides.combine(bstrides[2]);

	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    for(int i2=0; i2<I2; i2++){
	      r.arr[roffs+i0*r0+i1*r1+i2*r2]+=std::real(t);
	      r.arrc[roffs+i0*r0+i1*r1+i2*r2]+=std::imag(t);
	    }
	return;
      }

    }


  };

}
 
#endif 
