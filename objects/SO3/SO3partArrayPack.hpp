
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partArrayPack
#define _GElibSO3partArrayPack

#include "GElib_base.hpp"
#include "SO3partArrayPackView.hpp"
//#include "TensorPack.hpp"
#include "TensorPackVirtual.hpp"


namespace GElib{


  template<typename TYPE>
  class SO3partArrayPack: public cnine::TensorPackVirtual<complex<TYPE>, SO3partArrayPackView<TYPE> >{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::fill_zero fill_zero;

    typedef cnine::Gdims Gdims;
    typedef cnine::TensorPackDir TensorPackDir;
    typedef cnine::TensorPackVirtual<complex<TYPE>, SO3partArrayPackView<TYPE> > VirtualTensorPack;

    using VirtualTensorPack::VirtualTensorPack;
    using VirtualTensorPack::arr;
    using VirtualTensorPack::move_to_device;

    //using cnine::TensorPackVirtual<TYPE, SO3partArrayPackView<TYPE> >::TensorPackVirtual; 
    //using cnine::TensorPackVirtual<TYPE, SO3partArrayPackView<TYPE> >::arr; 
    //using cnine::TensorPackVirtual<TYPE, SO3partArrayPackView<TYPE> >::move_to_device;


    ~SO3partArrayPack(){
    }


    // ---- Constructors -------------------------------------------------------------------------------------


    SO3partArrayPack(const Gdims& _dims, const int l, const int n, const int N, const int _dev=0):
      SO3partArrayPack(TensorPackDir(_dims.cat({2*l+1,n}),N),_dev){}


    // ---- Named constructors -------------------------------------------------------------------------------

    
    static SO3partArrayPack<TYPE> zero(const int n, const Gdims& _dims, const int l, const int c, const int _dev=0){
      return SO3partArrayPack<TYPE>(TensorPackDir(_dims.cat({2*l+1,c}),n),cnine::fill_zero(),_dev);}
    
    static SO3partArrayPack<TYPE> sequential(const int n, const Gdims& _dims, const int l, const int c, const int _dev=0){
      return SO3partArrayPack<TYPE>(TensorPackDir(_dims.cat({2*l+1,c}),n),cnine::fill_sequential(),_dev);}
    
    static SO3partArrayPack<TYPE> gaussian(const int n, const Gdims& _dims, const int l, const int c, const int _dev=0){
      return SO3partArrayPack<TYPE>(TensorPackDir(_dims.cat({2*l+1,c}),n),cnine::fill_gaussian(),_dev);}
    

    // ---- Conversions ---------------------------------------------------------------------------------------


    SO3partArrayPack(const VirtualTensorPack& x):
      VirtualTensorPack(x){}


  public: // ---- Access --------------------------------------------------------------------------------------



  };

}

#endif 


    /*
    SO3partArrayPack(const TensorPackDir& _dir, const int _dev=0): 
      SO3partArrayPackView<TYPE>(_dir,cnine::MemArr<TYPE>(_dir.total(),_dev)){}

    SO3partArrayPack(const TensorPackDir& _dir, const cnine::fill_zero& dummy, const int _dev=0): 
      SO3partArrayPackView<TYPE>(_dir,cnine::MemArr<TYPE>(_dir.total(),dummy,_dev)){}
    
    SO3partArrayPack(const TensorPackDir& _dir, const cnine::fill_sequential& dummy, const int _dev=0):
      SO3partArrayPack(_dir,_dev){
      int N=_dir.total();
      for(int i=0; i<N; i++)
	arr[i]=i;
      move_to_device(_dev);
    }

    SO3partArrayPack(const TensorPackDir& _dir, const cnine::fill_gaussian& dummy, const int _dev=0):
      SO3partArrayPack(_dir,_dev){
      int N=_dir.total();
      normal_distribution<double> distr;
      for(int i=0; i<N; i++)
	arr[i]=distr(rndGen)*dummy.c;
      move_to_device(_dev);
    }
    */
    
