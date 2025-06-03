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

  template<typename TYPE>
  class SparseTensorView{
  public:

    typedef TensorView<TYPE> TENSOR;
    
    Gdims dims;

    Gdims ddims;
    GstridesB dstrides;
    int split=0;
    int sstride;

    shared_ptr<SparseIndexerBase> indexer;
    TENSOR mx;

    SparseTensorView(const Gdims& _dims, const int _split, const TensorView<int>& index_list, 
      const int fcode=0, const int _dev=0):
      dims(_dims),
      split(_split){

      CNINE_ASSRT(index_list.ndims()==2);
      int d=index_list.dims(1);

      switch(d){
      case 2:
	indexer.reset(new SparseIndexer2(index_list));
	break;
      default:
	CNINE_UNIMPL();
      }

      int nfilled=indexer->nfilled();
      ddims=dims.chunk(d);
      dstrides=split_strides(ddims,_split,nfilled);
      sstride=dstrides(split)*ddims(split);
      mx.reset(TENSOR(Gdims(nfilled,ddims),fcode,_dev));
      mx.strides=dstrides.insert(0,dstrides[split]*ddims[split]);
    }

//     SparseTensorView(const SparseIndexerBase& _indexer, const TENSOR& x):
//       indexer(_indexer), 
//       mx(x){
//       int nfi
//     }
      
    GstridesB split_strides(const Gdims& _ddims, const int split, const int nsparse){
      int k=_ddims.size();
      CNINE_ASSRT(k>0);
      GstridesB R(k,fill_raw());
      R[k-1]=1;
      for(int i=k-2; i>=0; i--){
	R[i]=R[i+1]*_ddims[i+1];
	if(i==split-1) R[i]*=nsparse;
      }
      return R;
    }
      

  public: // ---- Access -------------------------------------------------------------------------------------


    TENSOR dense_block(const int i0) const{
      return TENSOR(mx.arr+indexer->offset(i0)*sstride,ddims,dstrides);
    }

    TENSOR dense_block(const int i0, const int i1) const{
      return TENSOR(mx.arr+indexer->offset(i0,i1)*sstride,ddims,dstrides);
    }

    TENSOR dense_block(const int i0, const int i1, const int i2) const{
      return TENSOR(mx.arr+indexer->offset(i0,i1,i2)*sstride,ddims,dstrides);
    }

    TENSOR dense_block(const Gindex& ix) const{
      return TENSOR(mx.arr+indexer->offset(ix)*sstride,ddims,dstrides);
    }

    
  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_dense_block(std::function<void(const Gindex& ix, const TENSOR& T)> lambda) const{
      indexer->for_each([&](const Gindex& ix, const int v){
	  lambda(ix,dense_block(ix));});
    }

    void for_each_dense_block(std::function<void(const int i, const int j, const TENSOR& T)> lambda) const{
      indexer->for_each([&](const int i, const int j, const int v){
	  lambda(i,j,dense_block(i,j));});
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      ostringstream oss;
      for_each_dense_block([&](const Gindex& ix, const TENSOR& x){
	  oss<<indent<<"Block "<<ix<<":"<<endl;
	  oss<<x.str(indent+"  ");
	});
      return oss.str();
    }
    
  };


}

#endif 
