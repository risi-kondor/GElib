/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _Cnine_BlockCparseMatrix
#define _Cnine_BlockCparseMatrix

#include "Ltensor.hpp"
#include "map_of_maps.hpp"
#include "hlists.hpp"
#include "map_of_lists.hpp"
#include "double_indexed_map.hpp"
#include "once.hpp"
#include "GatherMapB.hpp"

namespace cnine{

  template<typename TYPE> class BlockCsparseMatrix;


#ifdef _WITH_CUDA
  template<typename TYPE>
  void BSM_apply_to_BV_cu(const BlockCsparseMatrix<TYPE>& x, const TensorView<TYPE>& r, const TensorView<TYPE>& y, const cudaStream_t& stream);
#endif
  
  template<typename TYPE>
  class BlockCsparseMatrix{
  public:
    
    typedef TensorView<TYPE> TENSOR;
    typedef TensorView<int> ITENSOR;

    int blockn;
    int blockm;
    int nblocks;
    int mblocks;
    
    TENSOR mx;
    //map_of_maps<int,int,int> offsets;
    double_indexed_map<int,int,int> offsets;
    
    once<GatherMapB> gather_mapL=once<GatherMapB>([this](){
	return GatherMapB(offsets.rmap);});

    once<GatherMapB> gather_mapR=once<GatherMapB>([this](){
	return GatherMapB(offsets.cmap);});

    once<TensorView<int> > row_offsets=once<TensorView<int> >([this](){
	TensorView<int> R({offsets.rmap.size()});
	int i=0;
	for(auto& p:offsets.rmap)
	  R.set(i++,p.second.begin()->second);
	return R;
      });

    RemoteCopy<int,ITENSOR> row_offsets_on_device=cnine::RemoteCopy<int,ITENSOR>([this](const int& _dev){
	return to_share(new ITENSOR(row_offsets(),_dev));});
    
      
  public: // ---- Constructors -----------------------------------------------------------------------------


    BlockCsparseMatrix(){}

    BlockCsparseMatrix(const int _nblocks, const int _mblocks, const int _blockn, const int _blockm):
      nblocks(_nblocks), mblocks(_mblocks), blockn(_blockn), blockm(_blockm){}

    BlockCsparseMatrix(const int _nblocks, const int _mblocks, const int _blockn, const int _blockm,
      const hlists<int>& mask, const int fcode, const int _dev=0):
      nblocks(_nblocks), mblocks(_mblocks), blockn(_blockn), blockm(_blockm),
      mx(cnine::dims(mask.total(),_blockn,_blockm),fcode,_dev){
      int t=0; 
      mask.for_each([&](const int i, const int j){offsets.set(i,j,t++);});
    }

    BlockCsparseMatrix(const int _nblocks, const int _mblocks, const int _blockn, const int _blockm,
      const map_of_lists<int,int>& mask, const int fcode, const int _dev=0):
      nblocks(_nblocks), mblocks(_mblocks), blockn(_blockn), blockm(_blockm),
      mx(cnine::dims(mask.total(),_blockn,_blockm),fcode,_dev){
      int t=0; 
      mask.for_each([&](const int i, const int j){offsets.set(i,j,t++);});
    }


  public: // ---- Named Constructors -----------------------------------------------------------------------


    //static BlockCsparseMatrix outer(const TENSOR& x, const TENSOR& y, const )


  public: // ---- Access -----------------------------------------------------------------------------------


    int get_dev()const{
      return mx.get_dev();
    }

    int n_blocks() const{
      return nblocks;
    }

    int m_blocks() const{
      return mblocks;
    }

    int block_n() const{
      return blockn;
    }

    int block_m() const{
      return blockm;
    }

    int nrows() const{
      return nblocks*blockn;
    }

    int ncols() const{
      return mblocks*blockm;
    }

    bool is_filled(const int i, const int j) const{
      if(offsets.rmap.find(i)==offsets.rmap.end()) return false;
      if(offsets.rmap[i].find(j)==offsets.rmap[i].end()) return false;
      return true;
    }
    
    int offset(const int i, const int j) const{
      CNINE_ASSRT(offsets.rmap.find(i)!=offsets.rmap.end());
      CNINE_ASSRT(offsets.rmap[i].find(j)!=offsets.rmap[i].end());
      return offsets.rmap[i][j];
    }

    TENSOR block(const int i, const int j) const{
      CNINE_ASSRT(i<nblocks);
      CNINE_ASSRT(j<mblocks);
      return mx.slice(0,offset(i,j));
    }

    void for_each_block(const std::function<void(const int, const int, const TENSOR&)>& lambda) const{
      offsets.for_each([&](const int i, const int j, const int offset){
	  lambda(i,j,mx.slice(0,offset));});
    }
      

  public: // ---- Operations --------------------------------------------------------------------------------


    TensorView<TYPE> operator*(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(x.dim(0)==mblocks*blockm);
      CNINE_ASSRT(get_dev()==x.get_dev());
      TensorView<TYPE> R({nblocks*blockn,x.dim(1)},0,get_dev());
      apply_to(R,x);
      return R;
    }

    void apply_to(const TensorView<TYPE>& r, const TensorView<TYPE>& x) const{
      int dev=get_dev();
      CNINE_ASSRT(x.get_dev()==dev);
      CNINE_ASSRT(r.get_dev()==dev);
      CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(x.dim(0)==mblocks*blockm);
      CNINE_ASSRT(r.ndims()==2);
      CNINE_ASSRT(r.dim(0)==nblocks*blockn);
      CNINE_ASSRT(x.dim(1)==r.dim(1));

      if(dev==0){
	for_each_block([&](const int i, const int j, const TENSOR& b){
	    r.rows(i*blockn,blockn).add_mprod(b,x.rows(j*blockm,blockm));
	  });
      }else{
	CUDA_STREAM(BSM_apply_to_BV_cu(*this,r,x,stream));
      }
    }


    void apply_transp_to(const TensorView<TYPE>& r, const TensorView<TYPE>& x)const{
      int dev=get_dev();
      CNINE_ASSRT(x.get_dev()==dev);
      CNINE_ASSRT(r.get_dev()==dev);
      CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(x.dim(0)==nblocks*blockn);
      CNINE_ASSRT(r.ndims()==2);
      CNINE_ASSRT(r.dim(0)==mblocks*blockm);
      CNINE_ASSRT(x.dim(1)==r.dim(1));

      if(dev==0){
	for_each_block([&](const int i, const int j, const TENSOR& b){
	    r.rows(j*blockm,blockm).add_mprod(b.transp(),x.rows(i*blockn,blockn));
	  });
      }else{
	CUDA_STREAM(BSM_apply_to_BV_cu(*this,r,x,stream));
      }
    }


    BlockCsparseMatrix operator*(const BlockCsparseMatrix& x){

      BlockCsparseMatrix R;
      int t=0;
      for(auto& p1:offsets.rmap)
	for(auto& p2:x.offsets.cmap){
	  bool found=0;
	  for(auto& q1:p1.second)
	    if(p2.second.find(q1.first)!=p2.second.end()){
	      found=true;
	      break;
	    }
	  if(found) R.offsets.set(p1.first,p2.first,t++);
	}

      R.mx.reset(TENSOR(dims(t,blockn,x.blockm),0,get_dev()));
      for(auto& p1:offsets.rmap)
	for(auto& p2:x.offsets.cmap)
	  for(auto& q1:p1.second)
	    if(p2.second.find(q1.first)!=p2.second.end())
	      R.block(p1.first,p2.first).add_mprod(block(p1.first,q1.first),x.block(q1.first,p2.first));
      return R;
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    static string classname(){
      return "BlockCsparseMatrix";
    }

    string repr() const{
      return "BlockCsparseMatrix";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for_each_block([&](const int i, const int j, const TENSOR& x){
	  oss<<indent<<"Block("<<i<<","<<j<<"):"<<endl;
	  oss<<x.to_string(indent+"  ");
	});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BlockCsparseMatrix& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif

