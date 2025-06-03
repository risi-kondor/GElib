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


#ifndef _CnineSparseIndexer2
#define _CnineSparseIndexer2

#include "Cnine_base.hpp"
#include "double_indexed_map.hpp"
#include "TensorView.hpp"
#include "SparseIndexerBase.hpp"

namespace cnine{

  class SparseIndexer2: public SparseIndexerBase{
  public:
    
    double_indexed_map<int,int,int> offsets;
    int _nfilled=0;

    SparseIndexer2(const TensorView<int>& list){
      CNINE_ASSRT(list.ndims()==2);
      int N=list.dim(0);
      for(int i=0; i<N; i++)
	offsets.set(list(i,0),list(i,1),_nfilled++);
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int dsparse() const{
      return 2;
    }

    int nfilled() const{
      return _nfilled;
    }

    int offset(const int i0, const int i1) const{
      return offsets(i0,i1);
    }

    int offset(const Gindex& x) const{
      CNINE_ASSRT(x.size()==2);
      return offsets(x[0],x[1]);
    }


  public: // ---- Lambdas -----------------------------------------------------------------------------------
    
    
    void for_each(std::function<void(const Gindex&, const int v)> lambda) const{
      offsets.for_each([&lambda](const int i0, const int i1, const int v){lambda(Gindex({i0,i1}),v);});
    }

    void for_each(std::function<void(const int i0, const int i1, const int v)> lambda) const{
      offsets.for_each([&lambda](const int i0, const int i1, const int v){lambda(i0,i1,v);});
    }


  };

}

#endif 


