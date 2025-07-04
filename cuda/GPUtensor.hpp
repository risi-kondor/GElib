#ifndef _GElibGPUtensor
#define _GElibGPUtensor

#include <cuda/std/array>
//#include "/usr/local/cuda-12.6/include/cuda/std/array"
#include "TensorView.hpp"


namespace cnine{

  //using namespace cnine;


  template<typename TYPE, int k>
  class GPUtensor{
  public:
    
    TYPE* arr;
    cuda::std::array<int,k> dims;
    cuda::std::array<int,k> strides;

    GPUtensor(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==k);
      arr=x.get_arr();
      for(int i=0; i<k; i++)
	dims[i]=x.dims[i];
      for(int i=0; i<k; i++)
	strides[i]=x.strides[i];
    }

    GPUtensor(const TensorView<complex<TYPE> >& x){
      CNINE_ASSRT(x.ndims()==k);
      arr=reinterpret_cast<TYPE*>(x.get_arr());
      for(int i=0; i<k; i++)
	dims[i]=x.dims[i];
      for(int i=0; i<k; i++)
	strides[i]=2*x.strides[i];
    }


  public: // ---- Access -----------------------------------------------------------------------------------------------


    __device__ TYPE operator()(const int i0) const{
      return *(arr+i0*strides[0]);
    }

    __device__ TYPE operator()(const int i0, const int i1) const{
      return *(arr+i0*strides[0]+i1*strides[1]);
    }

    __device__ TYPE operator()(const int i0, const int i1, const int i2) const{
      return *(arr+i0*strides[0]+i1*strides[1]+i2*strides[2]);
    }

    __device__ TYPE operator()(const int i0, const int i1, const int i2, const int i3) const{
      return *(arr+i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3]);
    }


    __device__ void set(const int i0, TYPE& v) const{
      *(arr+i0*strides[0])=v;
    }

    __device__ void set(const int i0, const int i1, TYPE& v) const{
      *(arr+i0*strides[0]+i1*strides[1])=v;
    }

    __device__ void set(const int i0, const int i1, const int i2, TYPE& v) const{
      *(arr+i0*strides[0]+i1*strides[1]+i2*strides[2])=v;
    }

    __device__ void set(const int i0, const int i1, const int i2, const int i3, TYPE& v) const{
      *(arr+i0*strides[0]+i1*strides[1]+i2*strides[2]+i3*strides[3])=v;
    }

    
  };


  /*
  template<typename TYPE, int k>
  class CGPUtensor{
  public:
    
    TYPE* arr;
    cuda::std::array<int,k> strides;

    CGPUtensor(const TensorView<complex<TYPE> >&  x){
      CNINE_ASSRT(x.ndims()==k);
      arr=reintrpret_cast<TYPE*>(x.get_arr());
      for(int i=0; i<k; i++)
	strides[i]=2*x.strides[i];
    }

  };
  */

}


#endif 
