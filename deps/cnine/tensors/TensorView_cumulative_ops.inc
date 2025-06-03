/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021--, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


void operator+=(const TYPE x) const {add(x);}
void operator+=(const TensorView& x) const{add(x);}
void operator-=(const TYPE x) const {add(-x);}
void operator-=(const TensorView& x) const {subtract(x);}


// ---- Addition ---------------------------------------------------------------------------------------------


void add(const TYPE x) const{
  if(asize()==0) return;
  if(x==0) return;
  auto R=scrunch();
  if(dev==0) R.for_each([&](TYPE& v){v+=x;});
  if(dev==1){
    if constexpr(std::is_same<TYPE,int>::value||
      std::is_same<TYPE,float>::value||
      std::is_same<TYPE,double>::value){
      CUDA_STREAM(TensorView_inc_cu(R,x,stream));
    }else{
      CNINE_CPUONLY();
    }
  }
}


void add(const TensorView& x) const{
  CNINE_DEVICE_SAME(x);
  CNINE_CHECK_SIZE(dims.check_eq(x.dims));
  TensorView_add(*this,x);
}


void add(const TensorView& x, const TYPE c) const{
  CNINE_DEVICE_SAME(x);
  CNINE_CHECK_SIZE(dims.check_eq(x.dims));
  CNINE_CPUONLY();
  assert(asize()==x.asize());
  if(dev==0){
    if(is_regular() && x.is_regular() && strides==x.strides){
      TYPE* ptr=const_cast<MemArr<TYPE>&>(arr).get_arr();/*+strides.offset*/
      TYPE* xptr=const_cast<MemArr<TYPE>&>(x.arr).get_arr()/*+x.strides.offset*/;
      for(size_t i=0; i<asize(); i++) ptr[i]+=c*xptr[i];
    }else{
      for_each([&](const Gindex& ix, TYPE& v){v+=c*x(ix);});
    }
  }
  if(dev==1){
    if(is_regular() && x.is_regular() && strides==x.strides){
      const TYPE alpha=c;
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize(), &alpha, x.arr, 1, arr, 1));
    }else
      CNINE_UNIMPL();
  }
}


// ---- Subtractions -----------------------------------------------------------------------------------------


//TensorView operator-(const TensorView& y){
//auto r=copy();
//r.subtract(y);
//return r;
//}


void subtract(const TensorView& x) const{
  CNINE_DEVICE_SAME(x);
  CNINE_CHECK_SIZE(dims.check_eq(x.dims));
  CNINE_CPUONLY();
  assert(asize()==x.asize());
  if(dev==0){
    if(is_regular() && x.is_regular() && strides==x.strides){
      TYPE* ptr=const_cast<MemArr<TYPE>&>(arr).get_arr();
      TYPE* xptr=const_cast<MemArr<TYPE>&>(x.arr).get_arr();
      for(size_t i=0; i<asize(); i++) ptr[i]-=xptr[i];
    }else
      for_each([&](const Gindex& ix, TYPE& v){v-=x(ix);});
  }
  if(dev==1){
    if(is_contiguous() && x.is_contiguous() && strides==x.strides){
      const float alpha=-1.0; // todo
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize(), &alpha, x.arr, 1, arr, 1));
    }else
      CNINE_UNIMPL();
  }
}


void subtract(const TensorView& x, const TYPE c) const{
  CNINE_DEVICE_SAME(x);
  CNINE_CHECK_SIZE(dims.check_eq(x.dims));
  CNINE_CPUONLY();
  assert(asize()==x.asize());
  if(dev==0){
    if(is_regular() && x.is_regular() && strides==x.strides){
      TYPE* ptr=const_cast<MemArr<TYPE>&>(arr).get_arr();/*+strides.offset*/
      TYPE* xptr=const_cast<MemArr<TYPE>&>(x.arr).get_arr()/*+x.strides.offset*/;
      for(size_t i=0; i<asize(); i++) ptr[i]-=c*xptr[i];
    }else{
      for_each([&](const Gindex& ix, TYPE& v){v-=c*x(ix);});
    }
  }
  if(dev==1){
    if(is_regular() && x.is_regular() && strides==x.strides){
      const TYPE alpha=c;
      //CUBLAS_SAFE(cublasSaxpy(cnine_cublas, asize(), &alpha, x.arr, 1, arr, 1));
    }else
      CNINE_UNIMPL();
  }
}


// ---- Other ------------------------------------------------------------------------------------------------


void add_sum(const int d, const TensorView& x) const{
  CNINE_ASSRT(x.dims.size()>d);
  for(int i=0; i<x.dims[d]; i++)
    add(x.slice(d,i));
}


void add_prod(const TensorView& x, const TensorView& y) const{
  CNINE_DEVICE_SAME(x);
  CNINE_DEVICE_SAME(y);
  CNINE_DIMS_SAME(x);
  CNINE_DIMS_SAME(y);
  if(dev==0){
    if(is_regular() && x.is_regular() && y.is_regular() && strides==x.strides&& strides==y.strides){
      TYPE* ptr=const_cast<MemArr<TYPE>&>(arr).get_arr();
      TYPE* xptr=const_cast<MemArr<TYPE>&>(x.arr).get_arr();
      TYPE* yptr=const_cast<MemArr<TYPE>&>(y.arr).get_arr();
      for(int i=0; i<asize(); i++) ptr[i]+=xptr[i]*yptr[i];
    }else
      for_each([&](const Gindex& ix, TYPE& v){v+=x(ix)*y(ix);});
  }
  if(dev==1){
    CNINE_UNIMPL();
  }
}


void add_broadcast(const int d, const TensorView& x) const{
  CNINE_ASSRT(d<dims.size());
  CNINE_CPUONLY();
  if(dev==0)
    for(int i=0; i<dims[d]; i++)
      slice(d)+=x;
}


void add_ReLU(const TensorView& x, const float alpha) const{
  CNINE_CHECK_SIZE(dims.check_eq(x.dims));
  assert(x.get_dev()==get_dev());
  int N=asize();
  if(dev==0){
    for(int i=0; i<N; i++) 
      arr[i]+=((x.arr[i]>0)+alpha*(x.arr[i]<0))*x.arr[i];
  }
  if(dev==1){
    flat_view().add_ReLU(x.flat_view(),alpha);
  }
}


void add_ReLU_back(const TensorView& g, const TensorView& x, const float alpha) const{
  CNINE_CHECK_SIZE(dims.check_eq(g.dims));
  assert(g.get_dev()==get_dev());
  int N=asize();
  if(dev==0){
    for(int i=0; i<N; i++)
      arr[i]+=((g.arr[i]>0)+alpha*(g.arr[i]<0))*g.arr[i];
  }
  if(dev==1){
    flat_view().add_ReLU_back(g.flat_view(),x.flat_view(),alpha);
  }
}
