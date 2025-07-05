/*
 * This file is part of GElib, a C++/CUDA library for group equivariant 
 * tensor operations. 
 *  
 * Copyright (c) 2025, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with GElib in the file NONCOMMERICAL.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in orginal
 * or modified form) must retain this copyright notice and must be 
 * accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GenericDiagCGproduct
#define _GenericDiagCGproduct


namespace GElib{


  template<typename GTYPE, typename... Args>
  typename std::enable_if<std::is_base_of<GtypeBase, GTYPE>::value, GTYPE>::type
  DiagCGproduct(const GTYPE& x, const GTYPE& y, const Args&... args){
    return x.DiagCGproduct(y,args...); 
  }


  // ---- Gparts ----------------------------------------------------------------------------------------------------


  template<typename GPART>
  typename std::enable_if<std::is_base_of<GpartBase, GPART>::value, GPART>::type
  DiagCGproduct(const GPART& x, const GPART& y, const typename GPART::IRREP_IX& ix){
    //auto& x=static_cast<const GPART&>(*this);
    int m=GPART::GROUP::CGmultiplicity(x.get_ix(),y.get_ix(),ix);
    GELIB_ASSRT(m>0);
    GELIB_ASSRT(x.getn()==y.getn());
    GPART R(ix,x.dominant_batch(y),x.dominant_gdims(y),x.getn(),0,x.dev);
    Gpart_add_DiagCGproduct(R,x,y);
    return R;
  }
  

  template<typename GPART>
  void Gpart_add_DiagCGproduct(GPART r, GPART x, GPART y, const int offs=0){
    const int dev=r.dev;
    GELIB_ASSRT(x.get_dev()==dev);
    GELIB_ASSRT(y.get_dev()==dev);
    if(!r.reconcile_batches(x,y)) GELIB_SKIP("batch dimensions cannot be reconciled.");
    if(!r.reconcile_grids(x,y)) GELIB_SKIP("grid dimensions cannot be reconciled.");
    r.co_canonicalize_to_5d(x,y);

    if(dev==0){
      auto C=r.get_CGmatrix(x,y);
      r.for_each_cell_multi(x,y,[&](const typename GPART::TENSOR& r, const typename GPART::TENSOR& x, const typename GPART::TENSOR& y){
			      GPART::add_DiagCGproduct_kernel(r,x,y,C,offs);});
    }

    if(dev==1){
      GPART::add_DiagCGproduct_dev(r,x,y,offs);
    }
  }


  template<typename GPART>
  void Gpart_add_DiagCGproduct_back0(GPART x, GPART r, GPART y, const int offs=0){
    const int dev=r.dev;
    GELIB_ASSRT(x.get_dev()==dev);
    GELIB_ASSRT(y.get_dev()==dev);
    if(!r.reconcile_batches(x,y)) GELIB_SKIP("batch dimensions cannot be reconciled.");
    if(!r.reconcile_grids(x,y)) GELIB_SKIP("grid dimensions cannot be reconciled.");
    r.co_canonicalize_to_5d(x,y);

    if(dev==0){
      auto C=r.get_CGmatrix(x,y);
      x.for_each_cell_multi(r,y,[&](const typename GPART::TENSOR& x, const typename GPART::TENSOR& r, const typename GPART::TENSOR& y){
			      GPART::add_DiagCGproduct_back0_kernel(r,x,y,C,offs);});
    }

    if(dev==1){
      GPART::add_DiagCGproduct_back0_dev(r,x,y,offs);
    }
  }


  template<typename GPART>
  void Gpart_add_DiagCGproduct_back1(GPART y, GPART r, GPART x, const int offs=0){
    const int dev=r.dev;
    GELIB_ASSRT(x.get_dev()==dev);
    GELIB_ASSRT(y.get_dev()==dev);
    if(!r.reconcile_batches(x,y)) GELIB_SKIP("batch dimensions cannot be reconciled.");
    if(!r.reconcile_grids(x,y)) GELIB_SKIP("grid dimensions cannot be reconciled.");
    r.co_canonicalize_to_5d(x,y);

    if(dev==0){
      auto C=r.get_CGmatrix(x,y);
      y.for_each_cell_multi(r,x,[&](const typename GPART::TENSOR& y, const typename GPART::TENSOR& r, const typename GPART::TENSOR& x){
			      GPART::add_DiagCGproduct_back1_kernel(r,x,y,C,offs);});
    }

    if(dev==1){
      GPART::add_DiagCGproduct_back1_dev(r,x,y,offs);
    }

  }


  // ---- Gvecs ------------------------------------------------------------------------------------------------------


  template<typename GVEC, typename... Args> 
  typename std::enable_if<std::is_base_of<GvecBase, GVEC>::value, GVEC>::type
  DiagCGproduct(const GVEC& x, const GVEC& y, const Args&... args){
    return x.DiagCGproduct(y,args...); 
  }


}

#endif 
