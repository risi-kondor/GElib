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


#ifndef _RtensorA_add_times_c_cop
#define _RtensorA_add_times_c_cop

#include "GenericCop.hpp"
#include "RtensorA.hpp"


namespace cnine{

  class RtensorArrayA;


  class RtensorA_add_times_c_cop: public Unary1Cop<RtensorA,RtensorArrayA,float >{
  public:

    RtensorA_add_times_c_cop(){}
    
    virtual void operator()(RtensorA& r, const RtensorA& y, const float& c) const{
      r.add(y,c);
    }

    template<typename IMAP>
    void operator()(const IMAP& map, RtensorArrayA& r) const{
      CNINE_UNIMPL();
    }

  };


  class RtensorA_add_div_c_cop: public Unary1Cop<RtensorA,RtensorArrayA,float >{
  public:

    RtensorA_add_div_c_cop(){}
    
    virtual void operator()(RtensorA& r, const RtensorA& y, const float& c) const{
      r.add(y,1.0/c);
    }

    template<typename IMAP>
    void operator()(const IMAP& map, RtensorArrayA& r) const{
      CNINE_UNIMPL();
    }

  };


  class RtensorA_inplace_times_c_cop: public Inplace1Cop<RtensorA,RtensorArrayA,float >{
  public:

    RtensorA_inplace_times_c_cop(){}
    
    virtual void operator()(RtensorA& r, const float& c) const{
      r.inplace_times(c);
    }

    template<typename IMAP>
    void operator()(const IMAP& map) const{
      CNINE_UNIMPL();
    }

  };


  class RtensorA_inplace_div_c_cop: public Inplace1Cop<RtensorA,RtensorArrayA,float >{
  public:

    RtensorA_inplace_div_c_cop(){}
    
    virtual void operator()(RtensorA& r, const float& c) const{
      r.inplace_times(c);
    }

    template<typename IMAP>
    void operator()(const IMAP& map) const{
      CNINE_UNIMPL();
    }

  };

}

#endif
