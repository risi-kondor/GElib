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


#ifdef _WITH_ATEN


TensorView(const at::Tensor& T):
  TensorView(copy(T)){}
  

static TensorView view(const at::Tensor& T){
  if(!((std::is_same<TYPE,int>::value && T.dtype()==torch::kInt32)||
      (std::is_same<TYPE,float>::value && T.dtype()==torch::kFloat32)||
      (std::is_same<TYPE,double>::value && T.dtype()==torch::kFloat64)||
      (std::is_same<TYPE,complex<float> >::value && T.dtype()==at::kComplexFloat)||
      (std::is_same<TYPE,complex<double> >::value && T.dtype()==at::kComplexDouble))){
    CNINE_ERROR("Cannot view a torch::Tensor with data type "+dtype_str(T)+" as a "+classname());
  }
  //if(T.dtype()==at::kComplexFloat) 
  if constexpr(std::is_same<TYPE,complex<float> >::value)
    return TensorView(MemArr<TYPE>(reinterpret_cast<complex<float>*>(T.data_ptr<c10::complex<float> >()),
	T.type().is_cuda()),Gdims(T),GstridesB(T));

  return TensorView(MemArr<TYPE>(T.data<TYPE>(),T.type().is_cuda()),Gdims(T),GstridesB(T));
}


static TensorView copy(const at::Tensor& T){
  auto _dims=Gdims(T);
  auto _strides=GstridesB(T);
  if(_strides.is_contiguous(_dims)){
    TensorView R(_dims,_strides,T.type().is_cuda());
    R=view(T);
    return R;
  }
  TensorView R(_dims,T.type().is_cuda());
  R=view(T);
  return R;
}


at::Tensor torch() const{
  // currently can only make regular tensors 

  auto options=torch::TensorOptions();

  if constexpr(std::is_same<TYPE,int>::value) options=options.dtype(torch::kInt32);
  if constexpr(std::is_same<TYPE,float>::value) options=options.dtype(torch::kFloat32);
  if constexpr(std::is_same<TYPE,double>::value) options=options.dtype(torch::kFloat64);
  if constexpr(std::is_same<TYPE,complex<float> >::value) options=options.dtype(at::kComplexFloat);
  if constexpr(std::is_same<TYPE,complex<double> >::value) options=options.dtype(at::kComplexDouble);

  if(dev==1) options=options.device(torch::kCUDA);
  else options=options.device(torch::kCPU);

  at::Tensor R(at::empty(dims.as_int64(),options));
  view(R)=*this;
  return R;
}


static string dtype_str(const at::Tensor& T){
  if(T.dtype()==torch::kInt32) return "kInt32";
  if(T.dtype()==torch::kFloat32) return "kFloat32";
  if(T.dtype()==torch::kFloat64) return "kFloat64";
  if(T.dtype()==at::kComplexFloat) return "kComplexFloat";
  if(T.dtype()==at::kComplexDouble) return "kComplexDouble";
  return "<unknown>";
}


#endif


//TensorView(const at::Tensor& T):
//  TensorView(Gdims(T),ifthen(is_contiguous(T),GstridesB(T),Gdims(T)),T.type().is_cuda()){
//  operator=(T);
//}

// TensorView& operator=(const at::Tensor& T){
//   CNINE_CONVERT_FROM_ATEN_WARNING();
//   CNINE_ASSRT(dims==Gdims(T));

//   if(!is_contiguous(T))

//   CNINE_ASSRT(dev==T.type().is_cuda());

//   if constexpr(std::is_same<TYPE,int>::value || std::is_same<TYPE,float>::value){
//     CNINE_ASSRT(strides==GstridesB(T));
//     if(dev==0){
//       std::copy(T.data<TYPE>(),T.data<TYPE>()+memsize(),reinterpret_cast<TYPE*>(arr.ptr()));
//     }
//     if(dev==1){
//       CUDA_SAFE(cudaMemcpy(arr.ptr(),T.data<TYPE>(),asize()*sizeof(TYPE),cudaMemcpyDeviceToDevice));
//     }
//     return *this;
//   }

//   if constexpr(std::is_same<TYPE,complex<float> >::value){
//     CNINE_ASSRT(strides==GstridesB(T));
//     if(dev==0){
//       std::copy(T.data<c10::complex<float>>(),T.data<c10::complex<float>>()+asize(),reinterpret_cast<c10::complex<float>*>(arr.ptr()));
//     }
//     if(dev==1){
//       CUDA_SAFE(cudaMemcpy(arr.ptr(),T.data<c10::complex<float>>(),asize()*sizeof(c10::complex<float>),cudaMemcpyDeviceToDevice));
//     }
//     return *this;
//   }

//   CNINE_UNIMPL();
//   return *this;
// }


// at::Tensor torch() const{
//   CNINE_CONVERT_TO_ATEN_WARNING();
//   //assert(dev==0);
//   int k=ndims();
//   //vector<int64_t> v(k); 
//   //for(int i=0; i<k; i++) v[i]=dims[i];

//   if constexpr(std::is_same<TYPE,int>::value){
//     if(dev==0){
//       at::Tensor R(at::zeros(dims.as_int64(),torch::CPU(at::kInt))); 
//       std::copy(arr.ptr(),arr.ptr()+dims.asize(),R.data<int>());
//       return R;
//     }
//   }

//   if constexpr(std::is_same<TYPE,float>::value){
//     if(dev==0){
//       at::Tensor R(at::zeros(dims.as_int64(),torch::CPU(at::kFloat))); 
//       std::copy(arr.ptr(),arr.ptr()+dims.asize(),R.data<float>());
//       return R;
//     }
//     if(dev==1){
//       at::Tensor R(at::zeros(dims.as_int64(),torch::CUDA(at::kFloat))); 
//       CUDA_SAFE(cudaMemcpy(R.data<float>(),arr.ptr(),dims.asize()*sizeof(float),cudaMemcpyDeviceToDevice));  
//       return R;
//     }
//   }

//   if constexpr(std::is_same<TYPE,complex<float> >::value){
//     if(dev==0){
//       at::Tensor R(at::zeros(dims.as_int64(),torch::CPU(at::kComplexFloat))); 
//       std::copy(arr.ptr(),arr.ptr()+dims.asize(),R.data<c10::complex<float>>());
//       return R;
//     }
//   }
      
//   CNINE_UNIMPL();
//   return at::Tensor(at::zeros(dims.as_int64(),torch::CPU(at::kFloat))); 
// }
//static bool is_contiguous(const at::Tensor& T){
//return GstridesB(T).is_contiguous(Gdims(T));
//}


