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


#ifndef _WorkStreamLoop
#define _WorkStreamLoop

#include "Cnine_base.hpp"


namespace cnine{

  extern thread_local DeviceSelector dev_selector;


  class WorkStreamLoop{
  public:
    
    //WorkStreamLoop(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, const cnine::Ctensor3_view y, const int offset, 
    //std::function<void(const cnine::Ctensor3_view r, const cnine::Ctensor3_view x, const cnine::Ctensor3_view y, const int offs)> lambda){
    //int Nblocks=dev_selector.max_mem/(x.n1*x.n2+y.n1*y.n2+r.n1);
    //}

  };


  class StreamingBlock{
  public:

    int old_dev=0;

    StreamingBlock(const int _dev):
      old_dev(dev_selector.dev){
      dev_selector.dev=_dev;
    }

    ~StreamingBlock(){
      dev_selector.dev=old_dev;
    }

  };

}

#endif 
