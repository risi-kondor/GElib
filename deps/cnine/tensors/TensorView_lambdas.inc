/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024--, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


// ---- Lambdas -----------------------------------------------------------------------------------------------


void for_each(std::function<void(TYPE&)> lambda) const{
  CNINE_CPUONLY();
  TYPE* arr=get_arr();
  switch(ndims()){
  case 0:
    return;
  case 1:
    {
      int n0=dims[0];
      int s0=strides[0];
      for(int i0=0; i0<n0; i0++)
	lambda(*(arr+i0*s0));
      return;
    }
    break;
  case 2:
    {
      int n0=dims[0];
      int s0=strides[0];
      int n1=dims[1];
      int s1=strides[1];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  lambda(*(arr+i0*s0+i1*s1));
      return;
    }
    break;
  case 3:
    {
      int n0=dims[0];
      int s0=strides[0];
      int n1=dims[1];
      int s1=strides[1];
      int n2=dims[2];
      int s2=strides[2];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    lambda(*(arr+i0*s0+i1*s1+i2*s2));
      return;
    }
    break;
  case 4:
    {
      int n0=dims[0];
      int s0=strides[0];
      int n1=dims[1];
      int s1=strides[1];
      int n2=dims[2];
      int s2=strides[2];
      int n3=dims[3];
      int s3=strides[3];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      lambda(*(arr+i0*s0+i1*s1+i2*s2+i3*s3));
      return;
    }
    break;
  case 5:
    {
      int n0=dims[0];
      int s0=strides[0];
      int n1=dims[1];
      int s1=strides[1];
      int n2=dims[2];
      int s2=strides[2];
      int n3=dims[3];
      int s3=strides[3];
      int n4=dims[4];
      int s4=strides[4];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      for(int i4=0; i4<n4; i4++)
		lambda(*(arr+i0*s0+i1*s1+i2*s2+i3*s3+i4*s4));
      return;
    }
    break;
  default:
    dims.for_each_index([&](const Gindex& ix){
	lambda(arr[strides.offs(ix)]);});
  }
}



void for_each(const std::function<void(const int i0, const int i1, TYPE& x)>& lambda) const{
  CNINE_ASSRT(ndims()==2);
  int n0=dims[0];
  int n1=dims[1];
  for(int i0=0; i0<n0; i0++)
    for(int i1=0; i1<n1; i1++)
      lambda(i0,i1,arr[strides.offs(i0,i1)]);
}

void for_each(const std::function<void(const Gindex&, TYPE& x)>& lambda) const{
  dims.for_each_index([&](const Gindex& ix){
      lambda(ix,const_cast<MemArr<TYPE>&>(arr)[strides.offs(ix)]);});
}


// ---- Zip ---------------------------------------------------------------------------------------------------


void zip(const TensorView& y, std::function<void(TYPE&,TYPE&)> lambda) const{
  CNINE_CPUONLY();
  CNINE_ASSRT(dims==y.dims);
  TYPE* xarr=get_arr();
  TYPE* yarr=y.get_arr();
  switch(ndims()){
  case 0:
    return;
  case 1:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      for(int i0=0; i0<n0; i0++)
	lambda(*(xarr+i0*s0x),*(yarr+i0*s0y));
    }
    break;
  case 2:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      int n1=dims[1];
      int s1x=strides[1];
      int s1y=y.strides[1];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  lambda(*(xarr+i0*s0x+i1*s1x),*(yarr+i0*s0y+i1*s1y));
    }
    break;
  case 3:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      int n1=dims[1];
      int s1x=strides[1];
      int s1y=y.strides[1];
      int n2=dims[2];
      int s2x=strides[2];
      int s2y=y.strides[2];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    lambda(*(xarr+i0*s0x+i1*s1x+i2*s2x),*(yarr+i0*s0y+i1*s1y+i2*s2y));
    }
    break;
  case 4:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      int n1=dims[1];
      int s1x=strides[1];
      int s1y=y.strides[1];
      int n2=dims[2];
      int s2x=strides[2];
      int s2y=y.strides[2];
      int n3=dims[3];
      int s3x=strides[3];
      int s3y=y.strides[3];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      lambda(*(xarr+i0*s0x+i1*s1x+i2*s2x+i3*s3x),*(yarr+i0*s0y+i1*s1y+i2*s2y+i3*s3y));
    }
    break;
  case 5:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      int n1=dims[1];
      int s1x=strides[1];
      int s1y=y.strides[1];
      int n2=dims[2];
      int s2x=strides[2];
      int s2y=y.strides[2];
      int n3=dims[3];
      int s3x=strides[3];
      int s3y=y.strides[3];
      int n4=dims[4];
      int s4x=strides[4];
      int s4y=y.strides[4];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      for(int i4=0; i4<n4; i4++)
		lambda(*(xarr+i0*s0x+i1*s1x+i2*s2x+i3*s3x+i4*s4x),*(yarr+i0*s0y+i1*s1y+i2*s2y+i3*s3y+i4*s4y));
    }
    break;
  default:
    dims.for_each_index([&](const Gindex& ix){
	lambda(arr[strides.offs(ix)],y.arr[y.strides.offs(ix)]);});
    }
}


// passing the lambda by reference prevents capture
void zip(const TensorView& y, TensorView& z, std::function<void(TYPE&,TYPE&,TYPE&)> lambda) const{
  CNINE_CPUONLY();
  CNINE_ASSRT(dims==y.dims);
  CNINE_ASSRT(dims==z.dims);
  TYPE* xarr=get_arr();
  TYPE* yarr=y.get_arr();
  TYPE* zarr=z.get_arr();
  switch(ndims()){
  case 0:
    return;
  case 1:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      int s0z=z.strides[0];
      for(int i0=0; i0<n0; i0++)
	lambda(*(xarr+i0*s0x),*(yarr+i0*s0y),*(zarr+i0*s0z));
    }
    break;
  case 2:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      int s0z=z.strides[0];
      int n1=dims[1];
      int s1x=strides[1];
      int s1y=y.strides[1];
      int s1z=z.strides[1];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
		lambda(*(xarr+i0*s0x+i1*s1x),
		  *(yarr+i0*s0y+i1*s1y),
		  *(zarr+i0*s0z+i1*s1z));
    }
    break;
  case 3:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      int s0z=z.strides[0];
      int n1=dims[1];
      int s1x=strides[1];
      int s1y=y.strides[1];
      int s1z=z.strides[1];
      int n2=dims[2];
      int s2x=strides[2];
      int s2y=y.strides[2];
      int s2z=z.strides[2];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
		lambda(*(xarr+i0*s0x+i1*s1x+i2*s2x),
		  *(yarr+i0*s0y+i1*s1y+i2*s2y),
		  *(zarr+i0*s0z+i1*s1z+i2*s2z));
    }
    break;
  case 4:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      int s0z=z.strides[0];
      int n1=dims[1];
      int s1x=strides[1];
      int s1y=y.strides[1];
      int s1z=z.strides[1];
      int n2=dims[2];
      int s2x=strides[2];
      int s2y=y.strides[2];
      int s2z=z.strides[2];
      int n3=dims[3];
      int s3x=strides[3];
      int s3y=y.strides[3];
      int s3z=z.strides[3];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
		lambda(*(xarr+i0*s0x+i1*s1x+i2*s2x+i3*s3x),
		  *(yarr+i0*s0y+i1*s1y+i2*s2y+i3*s3y),
		  *(zarr+i0*s0z+i1*s1z+i2*s2z+i3*s3z));
    }
    break;
  case 5:
    {
      int n0=dims[0];
      int s0x=strides[0];
      int s0y=y.strides[0];
      int s0z=z.strides[0];
      int n1=dims[1];
      int s1x=strides[1];
      int s1y=y.strides[1];
      int s1z=z.strides[1];
      int n2=dims[2];
      int s2x=strides[2];
      int s2y=y.strides[2];
      int s2z=z.strides[2];
      int n3=dims[3];
      int s3x=strides[3];
      int s3y=y.strides[3];
      int s3z=z.strides[3];
      int n4=dims[4];
      int s4x=strides[4];
      int s4y=y.strides[4];
      int s4z=z.strides[4];
      for(int i0=0; i0<n0; i0++)
	for(int i1=0; i1<n1; i1++)
	  for(int i2=0; i2<n2; i2++)
	    for(int i3=0; i3<n3; i3++)
	      for(int i4=0; i4<n4; i4++)
		lambda(*(xarr+i0*s0x+i1*s1x+i2*s2x+i3*s3x+i4*s4x),
		  *(yarr+i0*s0y+i1*s1y+i2*s2y+i3*s3y+i4*s4y),
		  *(zarr+i0*s0z+i1*s1z+i2*s2z+i3*s3z+i4*s4z));
    }
    break;
  default:
    dims.for_each_index([&](const Gindex& ix){
	lambda(arr[strides.offs(ix)],y.arr[y.strides.offs(ix)],z.arr[z.strides.offs(ix)]);});
  }
}


