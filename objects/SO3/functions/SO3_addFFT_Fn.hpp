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
#include "SO3FourierMatrixBank.hpp"

extern GElib::SO3FourierMatrixBank SO3FourierMxBank;


namespace GElib{


  class SO3part_addFFT_Fn{
  public:

    typedef cnine::CtensorB Ctensor;


    void operator()(const cnine::Ctensor3_view& p, const cnine::Ctensor4_view& f){
      int dev=p.dev;
      assert(f.dev==dev);

      assert(p.n1==p.n2);
      assert(p.n0==f.n0);
      int b=f.n0;
      int L=p.n1;
      int l=(L-1)/2;
      int Npsi=f.n3;
      int Ntheta=f.n2;
      int Nphi=f.n1;
      SO3FourierMatrixBank& bank=SO3FourierMxBank;

      //cout<<0<<endl;
      Ctensor A=Ctensor::zero(cnine::Gdims(b,Nphi,Ntheta,L),dev);
      A.view4().add_mix_3_0(f,bank.iFmatrix(l,Npsi,dev).view2());
      //cout<<1<<endl;

      Ctensor B=Ctensor::zero(cnine::Gdims(b,L,Ntheta,L),dev);
      B.view4().add_mix_1_0(A.view4(),bank.iFmatrix(l,Nphi,dev).view2());
      //cout<<2<<endl;

      B.view4().add_contract_abic_bic_abc_to(p,bank.Dmatrix(l,Ntheta,dev).view3());
      //cout<<3<<endl;
    }

  };

}

#endif 
