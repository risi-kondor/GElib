
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


#ifndef _MultiLoop
#define _MultiLoop
#include "ThreadGroup.hpp"

namespace cnine{

  extern thread_local int nthreads;


  class MultiLoop{
  public:
    
    MultiLoop(const int n, std::function<void(int)> lambda){

      if(nthreads<=1){
	for(int i=0; i<n; i++) lambda(i);
	return;
      }
      
      ThreadGroup threads(nthreads);
      for(int i=0; i<n; i++)
	threads.add(std::max(1,nthreads/n),lambda,i);

    }

  };

}

#endif 
