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

#include "Cnine_base.cpp"

#include "CnineSession.hpp"
#include "MultiLoop.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session(4);

  MultiLoop(10,[](const int i){
      COUT("Starting job  "<<i<<" with "<<nthreads<<" threads.");
      this_thread::sleep_for(chrono::milliseconds(uniform_int_distribution<int>(0,1000)(rndGen))); 
      COUT("Done with job "<<i);
    });

  

}
