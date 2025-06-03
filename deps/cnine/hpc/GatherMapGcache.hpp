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

#ifndef _GatherMapGcache
#define _GatherMapGcache

#include "Cnine_base.hpp"
#include "ptr_arg_indexed_cache.hpp"
#include "GatherMapB.hpp"
#include "compact_array_pool.hpp"


namespace cnine{

  class GatherMapGcache: public ptr_arg_indexed_cache<GatherMapB,int,shared_ptr<compact_array_pool<int> > >{
  public:

    typedef compact_array_pool<int> GMAP;
    typedef ptr_arg_indexed_cache<GatherMapB,int,shared_ptr<compact_array_pool<int> > > BASE;

    GatherMapGcache():
      BASE([](const GatherMapB& map, const int& dev){
	  return shared_ptr<GMAP>(new GMAP(map,dev));}){}


  public: // ---- Access -------------------------------------------------------------------------------------


    const GMAP& operator(const GatherMapB& map, const int dev){
      return *BASE::operator(map,dev);
    }

  };

}

#endif 
