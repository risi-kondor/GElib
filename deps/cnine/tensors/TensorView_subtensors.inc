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


// ---- Rows and columns ------------------------------------------------------------------------------------


TensorView row(const int i) const{
  CNINE_ASSRT(ndims()==2);
  CNINE_ASSRT(i<dims[0]);
  return TensorView(arr+strides[0]*i,{dims[1]},{strides[1]});
}

TensorView rows(const int i, const int n) const{
  CNINE_ASSRT(ndims()==2);
  CNINE_ASSRT(i+n<=dims[0]);
  return TensorView(arr+strides[0]*i,{n,dims[1]},{strides[0],strides[1]});
}

TensorView col(const int i) const{
  CNINE_ASSRT(ndims()==2);
  CNINE_ASSRT(i<dims[1]);
  return TensorView(arr+strides[1]*i,{dims[0]},{strides[0]});
}

TensorView cols(const int i, const int n) const{
  CNINE_ASSRT(ndims()==2);
  CNINE_ASSRT(i+n<=dims[1]);
  return TensorView(arr+strides[1]*i,{dims[0],n},{strides[0],strides[1]});
}

//TensorView columnss(const int i, const int n) const{
//CNINE_ASSRT(ndims()==2);
//CNINE_ASSRT(i+n<=dims[1]);
//return TensorView(arr+strides[1]*i,{dims[0],n},{strides[0],strides[1]});
//}

[[deprecated]]
tensor1_view<TYPE> rowv(const int i) const{
  CNINE_ASSRT(ndims()==2);
  CNINE_ASSRT(i<dims[0]);
  return tensor1_view<TYPE>(arr.ptr()+strides[0]*i,dims[1],strides[1]);
}


// ---- Blocks ---------------------------------------------------------------------------------------------


TensorView block(const Gdims& _dims) const{
  CNINE_ASSRT(_dims<=dims);
  return TensorView(arr,_dims,strides);
}

TensorView block(const Gdims& _dims, const Gindex& offs) const{
  CNINE_ASSRT(offs+_dims<=dims);
  //cout<<strides.offs(offs)<<" "<<_dims<<" "<<strides<<endl;
  return TensorView(arr+strides.offs(offs),_dims,strides);
}

TensorView block(const int i0, const int m0) const{
  CNINE_ASSRT(ndims()==1);
  return block(Gdims({m0}),Gindex({i0}));
}

TensorView bblock(const int i0, const int m0) const{
  CNINE_ASSRT(ndims()==1);
  return block(Gdims({m0}),Gindex({i0}));
}

TensorView block(const int i0, const int i1, const int m0, const int m1) const{
  CNINE_ASSRT(ndims()==2);
  return block(Gdims({m0,m1}),Gindex({i0,i1}));
}


// ---- Diagonals ------------------------------------------------------------------------------------------


TensorView diag() const{
  CNINE_ASSRT(ndims()==2);
  CNINE_ASSRT(dims[0]==dims[1]);
  return TensorView(arr,{dims[0]},{strides[0]+strides[1]});
}

TensorView diag(const vector<int>& ix) const{
  CNINE_ASSRT(ix.size()>0);
  CNINE_ASSRT(ix[0]<dims.size());
  int n=dims[ix[0]];
  int s=strides[ix[0]];
  for(int i=1; i<ix.size(); i++){
    CNINE_ASSRT(ix[i]<dims.size());
	CNINE_ASSRT(dims[ix[i]]==n);
	s+=strides[ix[i]];
  }
  vector<int> ix0(ix.begin()+1,ix.end());
  return TensorView(arr,dims.remove(ix0),strides.remove(ix0).set(ix[0],s));
}


// ---- Slices ---------------------------------------------------------------------------------------------


TensorView slice(const int d, const int i) const{
  CNINE_CHECK_RANGE(dims.check_in_range_d(d,i,string(__PRETTY_FUNCTION__)));
  return TensorView(arr+strides[d]*i,dims.remove(d),strides.remove(d)/*.inc_offset(strides[d]*i)*/);
}

TensorView slice(const Gindex& ix) const{
  const int k=ix.size();
  return TensorView(arr+strides.chunk(0,k)(ix),dims.chunk(k),strides.chunk(k));
}

TensorView slices(const int d, const int i, const int n) const{
  if(i+n>0) {CNINE_CHECK_RANGE(dims.check_in_range_d(d,i+n-1,string(__PRETTY_FUNCTION__)));}
  Gdims _dims(dims);
  return TensorView(arr+strides[d]*i,_dims.set(d,n),strides);
}

