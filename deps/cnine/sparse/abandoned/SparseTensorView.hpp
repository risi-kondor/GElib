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
#include "DSindexer.hpp"


namespace cnine{

  template<typename TYPE>
  class SparseTensorView{
  public:

    typedef TensorView<TYPE> TENSOR;
    
    Gdims dims;
    Gdims gdims;
    DSindexer indexer;
    //int _ngrid=0;
    TENSOR mx;

    SparseTensorView(const Gdims& _dims, const int _ngrid, const TensorView<int>& index_list, 
      const int fcode=0, const int _dev=0):
      dims(_dims),
      gdims(_dims.chunk(0,_ngrid)),
      DSindexer(_dims,_ngrid,index_list){
      mx.reset(TENSOR(Gdims(gdims,indexer.n_sparse_blocks(),indexer.cdims),fcode,_dev));
    }

    SparseTensorView(const DSindexer& _indexer, const TENSOR& x):
      indexer(_indexer), 
      mx(x){
    }
      

  public: // ---- Access -------------------------------------------------------------------------------------


    SparseTensorView block(const Gindex& gix) const{
      return SparseTensorView(mx.arr+gstrides(gix),indexer);
    }

    TENSOR block(const Gindex& gix, const Gindex& six) const{
      return TENSOR(mx.arr+indexer.block_offset(gix,six),indexer.cdims,indexer.cstrides);
    }

    
  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string str(const indent="") const{
      ostringstream oss;
      if(ngrid>0){
	indexer.gdims.for_each(const vector<int>& ix){
	  
	});
      }else{
      }
      return oss.str();
    }
    
  };


}

#endif 
