/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineMemoryManager
#define _CnineMemoryManager

#include "Cnine_base.hpp"

namespace cnine{

  class MemoryManager{
  public:

    virtual ~MemoryManager(){};
    virtual size_t size() const=0;
    virtual void* malloc(const int n) const=0;
    virtual void free(void* p) const=0;
    virtual void clear() const=0;

  };


  extern thread_local MemoryManager* vram_manager;

  class using_vram_manager{
  public:
    MemoryManager* old;
    using_vram_manager(MemoryManager* mm){
      old=vram_manager;
      vram_manager=mm;
    }
    ~using_vram_manager(){
      vram_manager=old;
    }
  };

}

#endif 
