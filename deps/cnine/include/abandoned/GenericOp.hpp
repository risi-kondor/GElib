/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _GenericOp
#define _GenericOp

#include "CtensorArrayA.hpp"

namespace cnine{

  class CtensorArrayA;

  //inline CtensorArrayA* arrayof(const CtensorA&){
  //return new CtensorArrayA(Gdims({}),Gdims({}));
  //}


  template<typename OBJ>
  class InplaceOp{
  public:
    virtual void operator()(OBJ& R) const=0;
  };


  template<typename OBJ, typename OBJ0>
  class UnaryOp{
  public:
    virtual void operator()(OBJ& R, const OBJ0& x0) const=0;
  };


  //template<typename OBJ>
  //class ARRAY;

  //template<>
  //typedef CtensorArrayA ARRAY<CtensorA>;

  //using ARRAY<CtensorA> =CtensorArrayA;

  template<typename OBJ, typename OBJ0, typename OBJ1>
  class BinaryOp{
  public:

    //typedef decltype(*OBJ().array_type()) OBJARR;
    //typedef decltype(*OBJ0().const_array_type()) COBJARR0;
    //typedef decltype(*OBJ1().const_array_type()) COBJARR1;

    virtual void operator()(OBJ& R, const OBJ0& x0, const OBJ1& x1) const=0;

    virtual void execG(CtensorArrayA& r, const CtensorArrayA& x, const CtensorArrayA& y,
      const int rn, const int xn, const int yn, const int rs, const int rx, const int ry) const{
      CNINE_UNIMPL();
    }

  };


}

#endif


    //virtual void map(cnine::CtensorArrayA& R, const cnine::CtensorArrayA& x0, const cnine::CtensorArrayA& x1) const{
