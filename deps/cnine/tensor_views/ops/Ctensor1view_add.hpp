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


#ifndef _Ctensor1view_add
#define _Ctensor1view_add

//#include "GenericCop.hpp"


namespace cnine{

#ifdef _WITH_CUDA
  template<typename CMAP>
  void Ctensor1view_add_cu(const CMAP& map, Ctensor1_view& r, const Ctensor1_view& x, const cudaStream_t& stream);
#endif 


  class Ctensor1view_add{
  public:

    Ctensor1view_add(){}
    
    void apply(Ctensor1_view& r, const Ctensor1_view& x, const float v, const bool add_flag=true) const{
      r.add(x,v);
    }

    template<typename IMAP>
    void operator()(const IMAP& map, Ctensor2_view& r, const Ctensor2_view& x, const bool add_flag=true) const{

      if(r.dev==0){
	// TODO 
      }

      if(r.dev==1){
#ifdef _WITH_CUDA
	cudaStream_t stream;
	CUDA_SAFE(cudaStreamCreate(&stream));
	Ctensor1view_add_cu(map,r,x,stream);
	CUDA_SAFE(cudaStreamSynchronize(stream));
	CUDA_SAFE(cudaStreamDestroy(stream));
#else
	CNINE_NOCUDA_ERROR;
#endif
      }
    }

  };


}

#endif
