// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef _SO3part_addRCGproduct_back0Fn
#define _SO3part_addRCGproduct_back0Fn

#include "GElib_base.hpp"
#include "SO3part2_view.hpp"
#include "CellFnTemplates.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;


namespace GElib{

  #ifdef _WITH_CUDA
  void SO3partB_addCGproduct_back0_cu(const cnine::Ctensor3_view& xg, cnine::Ctensor3_view rg, const cnine::Ctensor3_view& y, 
    const int offs, const cudaStream_t& stream);
  #endif


  class SO3part_addRCGproduct_back0Fn{
  public:

    typedef cnine::TensorView<complex<float> > TENSOR;

    void operator()(const TENSOR& _xg, const TENSOR& _g, const TENSOR& _y, const int _offs=0){

      CNINE_ASSRT(_xg.ndims()==5);
      CNINE_ASSRT(_g.ndims()==5);
      CNINE_ASSRT(_y.ndims()==5);

      const int l=(_g.dim(3)-1)/2;
      const int l1=(_xg.dim(3)-1)/2;
      const int l2=(_y.dim(3)-1)/2;
 
      const int N1=_xg.dim(4);
      const int N2=_y.dim(4);
      const int dev=_g.dev;

      assert(_offs+N1*N2<=_g.dim(4));
      assert(l>=abs(l1-l2) && l<=l1+l2);

      //LoggedTimer timer("  CGproductBack0("+to_string(l1)+","+to_string(l2)+","+to_string(l)+
      //")[b="+to_string(B)+",n1="+to_string(N1)+",n2="+to_string(N2)+",dev="+to_string(dev)+"]",B*(2*l1+1)*(2*l2+1)*N1*N2);

      if(dev==0){
	
	auto& C=SO3_cgbank.getf(CGindex(l1,l2,l));
	cnine::batched_mprod<TENSOR>(_xg,_g,_y,
	  [&](const TENSOR& _xg, const TENSOR& _g, const TENSOR& _y){
	    SO3part2_view xg(_xg);
	    SO3part2_view g(_g);
	    SO3part2_view y(_y);
	    int offs=_offs;
	    for(int n1=0; n1<N1; n1++){
	      for(int n2=0; n2<N2; n2++){
		for(int m1=-l1; m1<=l1; m1++){
		  for(int m2=std::max(-l2,-l-m1); m2<=std::min(l2,l-m1); m2++){
		    xg.inc(m1,n1,C(m1+l1,m2+l2)*g(m1+m2,offs+n2)*std::conj(y(m2,n2)));
		  }
		}
	      }
	      offs+=N2;
	    }
	  });

      }
      else{
	//CUDA_STREAM(SO3partB_addCGproduct_back0_cu(_xg,_g,_y,_offs,stream));
      }

    }

  };





}

#endif
