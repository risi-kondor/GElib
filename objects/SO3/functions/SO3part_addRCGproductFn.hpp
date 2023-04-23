// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3part_addRCGproductFn
#define _SO3part_addRCGproductFn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "SO3part2_view.hpp"
//#include "SO3part4_view.hpp"
#include "MultiLoop.hpp"
#include "GElibTimer.hpp"
#include "CellFnTemplates.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

  #ifdef _WITH_CUDA
  void SO3partB_addCGproduct_cu(cnine::Ctensor3_view r, const cnine::Ctensor3_view& x, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream);
  #endif 


  class SO3part_addRCGproductFn{
  public:

    typedef cnine::TensorView<complex<float> > TENSOR;

    void operator()(const TENSOR& _r, const TENSOR& _x, const TENSOR& _y, const int _offs=0){

      CNINE_ASSRT(_r.ndims()==5);
      CNINE_ASSRT(_x.ndims()==5);
      CNINE_ASSRT(_y.ndims()==5);

      const int l=(_r.dim(3)-1)/2;
      const int l1=(_x.dim(3)-1)/2;
      const int l2=(_y.dim(3)-1)/2;
 
      const int N1=_x.dim(4);
      const int N2=_y.dim(4);
      //const int B=_x.n0;
      //const int J=_x.n1;
      //const int K=_x.n2;
      const int dev=_r.dev;

      //CNINE_CHECK_DEV3(_r,_x,_y);
      //CNINE_CHECK_BATCH3(_r,_x,_y);
      //CNINE_ASSRT(_x.n1==_r.n1);
      //CNINE_ASSRT(_x.n2==_y.n1);
      CNINE_ASSRT(_offs+N1*N2<=_r.dim(4));
      CNINE_ASSRT(l>=abs(l1-l2) && l<=l1+l2);

      //LoggedTimer timer("  ReducingCGproduct("+to_string(l1)+","+to_string(l2)+","+to_string(l)+")[b="+
      //to_string(B)+", K="+to_string(K)+",n1="+to_string(N1)+",n2="+to_string(N2)+",dev="+to_string(dev)+"]",B*K*(2*l1+1)*(2*l2+1)*N1*N2);

      if(dev==0){

	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	cnine::batched_mprod<TENSOR>(_r,_x,_y,
	  [&](const TENSOR& _r, const TENSOR& _x, const TENSOR& _y){
	    SO3part2_view r(_r);
	    SO3part2_view x(_x);
	    SO3part2_view y(_y);

	    int offs=_offs;
	    for(int n1=0; n1<N1; n1++){
	      for(int n2=0; n2<N2; n2++){
		for(int m1=-l1; m1<=l1; m1++){
		  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		    r.inc(m1+m2,offs+n2,C(m1+l1,m2+l2)*x(m1,n1)*y(m2,n2));
		  }
		}
	      }
	      offs+=N2;
	    }
	  });

      }
      //else CUDA_STREAM(SO3partB_addReducingCGproduct_cu(_r,_x,_y,_offs,stream));

    }

  };

}

#endif 
