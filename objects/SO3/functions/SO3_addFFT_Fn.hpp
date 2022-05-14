// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3_addFFT_Fn
#define _SO3_addFFT_Fn

#include "GElib_base.hpp"
#include "CtensorB.hpp"
#include "Ctensor4_view.hpp"
#include "MultiLoop.hpp"

extern GElib::SO2FourierMatrixBank SO2FmatrixBank;


namespace GElib{


  class SO3part_addFFT_Fn{
  public:

    typedef cnine::CtensorB Ctensor;


    void operator()(const cnine::Ctensor3_view& p, const cnine::Ctensor4_view& f){

      assert(p.n1==p.n2);
      assert(p.n0==f.n0);
      int b=f.n0;

      Ctensor F0(cnine::Gdims(f.n3,p.n2));
      Ctensor F1(cnine::Gdims(f.n1,p.n1));
      Ctensor D(cnine::Gdims(f.n2,p.n0,p.n1));

      Ctensor A=Ctensor::zero(cnine::Gdims(b,f.n1,f.n2,p.n2));
      A.view4().add_mix_3_0(f,F0.view2());
      Ctensor B=Ctensor::zero(cnine::Gdims(b,p.n1,f.n2,p.n2));
      B.view4().add_mix_1_0(A.view4(),F1.view2());
      B.view4().add_contract_abic_bic_abc_to(p,D.view3());

    }

  };

}

#endif 
