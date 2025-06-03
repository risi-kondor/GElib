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


TYPE max() const{
  if(asize()==0) return 0;
  TYPE t=arr[0];
  if(is_contiguous()){
    for(size_t i=0; i<asize(); i++)
      if(arr[i]>t) t=arr[i];
  }else{
    for_each([&](const Gindex& ix, TYPE& v){
	if(v>t) t=v;});
  }
  return t; 
}

TYPE min() const{
  if(asize()==0) return 0;
  TYPE t=arr[0];
  if(is_contiguous()){
    for(size_t i=0; i<asize(); i++)
      if(arr[i]<t) t=arr[i];
  }else{
    for_each([&](const Gindex& ix, TYPE& v){
	if(v<t) t=v;});
  }
  return t; 
}

auto max_abs() const -> decltype(std::real(min())){
  if(asize()==0) return 0;
  decltype(std::real(min())) t=std::real(arr[0]);
  if(is_contiguous()){
    for(int i=0; i<asize(); i++)
      if(abs(arr[i])>t) t=abs(arr[i]);
  }else{
    //CNINE_UNIMPL()
    for_each([&](const Gindex& ix, TYPE& v){
	if(abs(v)>t) t=abs(v);});
  }
  return t; 
}

TYPE inp(const TensorView& y) const{
  CNINE_CPUONLY();
  CNINE_ASSRT(dims==y.dims);
  TYPE t=0;
  if(dev==0){
    if(is_regular() && y.is_regular()){
      for(size_t i=0; i<asize(); i++)
	t+=arr[i]*y.arr[i];
      //t+=std::conj(arr[i])*y.arr[i];
    }else{
      for_each([&](const Gindex& ix, TYPE& v){
	  t+=v*y(ix);});
      //t+=std::conj(v)*y(ix);});
    }
  }
  return t;
}

TYPE norm2() const{
  CNINE_CPUONLY();
  TYPE t=0;
  if(dev==0){
    if(is_contiguous()){
      for(size_t i=0; i<asize(); i++)
	t+=arr[i]*arr[i];
      //t+=std::conj(arr[i])*arr[i];
    }else{
      for_each([&](const Gindex& ix, TYPE& v){
	  t+=v*v;});
      //t+=std::conj(v)*v;});
    }
  }
  return t;
}

TYPE norm() const{
  return sqrt(norm2());
}

TYPE sum() const{
  if(asize()==0) return 0;
  TYPE t=0;
  if(is_contiguous()){
    for(size_t i=0; i<asize(); i++)
      t+=arr[i];
  }else{
    for_each([&](const Gindex& ix, TYPE& v){
	t+=v;});
  }
  return t; 
}





TYPE diff2(const TensorView& x) const{
  CNINE_ASSRT(x.asize()==asize());
  if(get_dev()==0 && x.get_dev()>0)
    return diff2(TensorView(x,0));
  TYPE t=0;
  if(is_regular() && x.is_regular()){
    for(size_t i=0; i<asize(); i++){
      const TYPE a=x.arr[i]-arr[i];
      if constexpr(is_complex<TYPE>())
	t+=a*std::conj(a);
      else
	t+=a*a;
    }
  }else{
    for_each([&](const Gindex& ix, TYPE& v){
	const TYPE a=x(ix)-(*this)(ix);
	if constexpr(is_complex<TYPE>())
	  t+=a*std::conj(a);
	else
	  t+=a*a;
      });
  }
  return t;
}

TYPE unitary_error() const{
  CNINE_ASSRT(dims[0]==dims[1]);
  TensorView A({dims[0],dims[0]},fill_zero());
  A.add_mprod(*this,transp());
  for(int i=0; i<dims[0]; i++)
    A.inc(i,i,-1.0);
  return A.norm();
}


