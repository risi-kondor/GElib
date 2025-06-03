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


#ifndef _GenericCop
#define _GenericCop

//#include "CtensorArrayA.hpp"


namespace cnine{


  template<typename OBJ, typename ARR>
  class InplaceCop{
  public:
    virtual void operator()(OBJ& R) const=0;
  };

  template<typename OBJ, typename ARR, typename ARG0>
  class Inplace1Cop{
  public:
    virtual void operator()(OBJ& R, const ARG0& arg0) const=0;
  };


  // ---- Unary 


  template<typename OBJ, typename ARR>
  class UnaryCop{
  public:
    virtual void operator()(OBJ& R, const OBJ& x0) const=0;
  };

  template<typename OBJ, typename ARR, typename ARG0>
  class Unary1Cop{
  public:
    virtual void operator()(OBJ& R, const OBJ& x0, const ARG0& arg0) const=0;
  };


  // ---- Binary 


  template<typename OBJ, typename ARR>
  class BinaryCop{
  public:

    virtual void operator()(OBJ& R, const OBJ& x0, const OBJ& x1) const=0;
    
  };

  template<typename OBJ, typename ARR, typename ARG0>
  class Binary1Cop{
  public:

    virtual void operator()(OBJ& R, const OBJ& x0, const OBJ& x1, const ARG0& arg0) const=0;
    
  };


}

#endif


    //virtual void map(cnine::CtensorArrayA& R, const cnine::CtensorArrayA& x0, const cnine::CtensorArrayA& x1) const{


    //virtual void operator()(ARR& r, const ARR& x, const ARR& y, const int mode) const{
    //GELIB_UNIMPL();
    //}

    //virtual void operator()(ARR& r, const ARR& x, const ARR& y, const BinaryImapBase& map) const{
    //GELIB_UNIMPL();
    //}
