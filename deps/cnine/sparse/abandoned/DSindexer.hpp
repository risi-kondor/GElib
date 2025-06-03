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


#ifndef _CnineSparseTensorView
#define _CnineSparseTensorView

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "SparseIndexerBase.hpp"
#include "SparseIndexer2.hpp"

namespace cnine{

  class DSindexer{
  public:

    Gdims dims;
    //Gdims gdims;
    Gdims sdims;
    Gdims cdims;
    //GstridesB gstrides;
    GstridesB cstrides;
    int sstride=0;
    int ngrid=0;

    shared_ptr<SparseIndexerBase> sparse_indexer;

    DSindexer(const Gdims& _dims, const TensorView<int>& index_list):
      dims(_dims){
      //gdims(_dims.chunk(0,_ngrid)),
      //sdims(_dims.chunk(0,index_list.dim(1))),
      //cdims(_dims.chunk(index_list.dim(1))){
      CNINE_ASSRT(index_list.ndims()==2);
      int d=index_list.dims(1)();
      if(d==2){
	sparse_indexer.reset(new SparseIndexer2(index_list));
      }
      gdims=dims.chunk(0,d);
      cdims=dims.chunk(d);
      sstride=cdims.asize();
      cstrides=GstridesB(cdims);
    }


    /*
    DSindexer(const Gdims& _dims, const int _ngrid, const TensorView<int>& index_list):
      dims(_dims),
      gdims(_dims.chunk(0,_ngrid)),
      sdims(_dims.chunk(_ngrid,index_list.dim(1))),
      cdims(_dims.chunk(_ngrid+index_list.dim(1))),
      ngrid(_ngrid){
      int d=index_list.dims(1)();
      CNINE_ASSRT(index_list.ndims()==2);
      CNINE_ASSRT(index_list.dims(1)==d);
      if(d==2){
	indexer.reset(new SparseIndexer2(index_list));
      }
      cstrides=GstridesB(cdims);
      sstride=cdims.asize();
      gstrides=GstridesB(gdims,sstride);
    }
    */


  public: // ---- Access -------------------------------------------------------------------------------------


    int n_blocks() const{
      return indexer.nfilled();
    }

    int block_offset(const Gindex& sindex){
      return sparse_indexer.offset(sindex);
    }

  };

}

#endif 

