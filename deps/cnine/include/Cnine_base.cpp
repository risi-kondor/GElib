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

#ifndef _Cnine_base_cpp
#define _Cnine_base_cpp

#include "Cnine_base.hpp"
#include "Primes.hpp"
#include "Factorial.hpp"
#include "FFactorial.hpp"
#include "DeltaFactor.hpp"
#include "CnineLog.hpp"
#include "CnineCallStack.hpp"
#include "GPUbuffer.hpp"
#include "AsyncGPUbuffer.hpp"
#include "MemoryManager.hpp"

#ifdef _WITH_CENGINE
#include "Cengine_base.cpp"
#endif 

namespace cnine{

  thread_local int nthreads=1;

  int streaming_footprint=1024;
  thread_local DeviceSelector dev_selector;

  thread_local MemoryManager* vram_manager=nullptr;

  string base_indent="";
  float* cuda_oneS=nullptr;
  Primes primes;
  Factorial factorial;
  FFactorial ffactorial;
  DeltaFactor delta_factor;

  CnineLog cnine_log;
  CallStack call_stack;

  AsyncGPUbuffer<int>  GatherRowsMulti_ibuf;
  AsyncGPUbuffer<int*>  GatherRowsMulti_ipbuf;
  GPUbuffer<float>  GatherRowsMulti_fbuf;

}

std::default_random_engine rndGen;

mutex cnine::CoutLock::mx;


#endif 
