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

#ifndef _cnine_gather_rows
#define _cnine_gather_rows

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "GatherMapPack.hpp"
#include "WeightedGatherMapB.hpp"
#include "FixedkGatherMap.hpp"
#include "Ltensor.hpp"
#include "logged_timer.hpp"
#include "MultiLoop.hpp"


namespace cnine{

#ifdef _WITH_CUDA
  extern void gatherRows_cu(const Rtensor2_view& r, const Rtensor2_view& x, const GatherMapB& g, const cudaStream_t& stream);
  extern void gatherRowsw_cu(const Rtensor2_view& r, const Rtensor2_view& x, const WeightedGatherMapB& g, const cudaStream_t& stream);
  extern void gatherRows_cu(const Rtensor2_view& r, const Rtensor2_view& x, const FixedkGatherMap& g, const cudaStream_t& stream);
  extern void gatherRowsMulti_cu(const Rtensor2_view& r, const Rtensor2_view& x, const vector<shared_ptr<const GatherMapB> >& maps, 
    const Ltensor<int>& out_offsets, const Ltensor<int>& in_offsets,const cudaStream_t& stream);
#endif 
  
  class GatherRows{
  public:

    template<typename TYPE>
    void operator()(const Ltensor<TYPE>& _r, const Ltensor<TYPE>& _x, const GatherMapB& g){
      CNINE_ASSRT(_r.ndims()==2);
      CNINE_ASSRT(_r.dim(1)%g.out_columns==0);
      CNINE_ASSRT(_x.ndims()==2);
      CNINE_ASSRT(_x.dim(1)%g.in_columns==0);
      int dev=_x.get_dev();
      CNINE_ASSRT(_r.get_dev()==dev);

      if(g.fixedk_maps.size()>0){
	for(auto& p: g.fixedk_maps)
	  (*this)(_r,_x,*p);
      }

      if(dynamic_cast<const WeightedGatherMapB*>(&g)) 
	weighted(_r,_x,dynamic_cast<const WeightedGatherMapB&>(g));

      if(g.size()==0) return;

      auto r=_r.view2();
      r.n0*=g.out_columns;
      r.n1=r.n1*g.out_columns_n/g.out_columns;
      r.s0/=g.out_columns;
      auto x=_x.view2();
      x.n0*=g.in_columns;
      x.n1=x.n1*g.in_columns_n/g.in_columns;
      x.s0/=g.in_columns;
      CNINE_ASSRT(r.n1==x.n1);
    
      if(_r.get_dev()==0){
	fnlog timer("GatherRows::operator()");
	//logged_timer ptimer("GatherRows(CPU)",r,x,((long long)g.n_ops())*x.n1);
	CNINE_ASSRT(g.get_dev()==0);
	int N=g.size();
	for(int i=0; i<N; i++){
	  auto targt=r.slice0(g.target(i));
	  int M=g.size_of(i);
	  for(int j=0; j<M; j++){
	    targt+=x.slice0(g(i,j));
	  }
	}
      }

      if(_r.get_dev()==1){
	g.sort();
	fnlog timer("GatherRows::operator()(G)");
	//logged_timer ptimer("GatherRows(GPU)",r,x,((long long)g.n_ops())*x.n1);
	CUDA_STREAM(gatherRows_cu(r,x,g,stream));
      }
    }


    template<typename TYPE>
    void operator()(const Ltensor<TYPE>& _r, const Ltensor<TYPE>& _x, const GatherMapPack& gmaps){
      CNINE_ASSRT(_r.ndims()==2);
      CNINE_ASSRT(_x.ndims()==2);
      CNINE_ASSRT(_r.dim(1)%gmaps.out_columns==0);
      CNINE_ASSRT(_x.dim(1)%gmaps.in_columns==0);
      int dev=_x.get_dev();
      CNINE_ASSRT(_r.get_dev()==dev);
      if(gmaps.size()==0) return;

      auto r=_r.view2();
      r.n0*=gmaps.out_columns;
      r.n1=r.n1*gmaps.out_columns_n/gmaps.out_columns;
      r.s0/=gmaps.out_columns;
      auto x=_x.view2();
      x.n0*=gmaps.in_columns;
      x.n1=x.n1*gmaps.in_columns_n/gmaps.in_columns;
      x.s0/=gmaps.in_columns;
      CNINE_ASSRT(r.n1==x.n1);
    
      if(dev==0){
	fnlog timer("GatherRows::operator_pack()");
	MultiLoop(gmaps.size(),[&](const int i){
	    GatherRows()(Ltensor<TYPE>(_r.rows(gmaps.out_offsets[i],gmaps[i].n_out)),
	      Ltensor<TYPE>(_x.rows(gmaps.in_offsets[i],gmaps[i].n_in)),gmaps[i]);});
      }

      if(dev==1){
	gmaps.sort();
	fnlog timer("GatherRows::operator_pack()(G)");
	//logged_timer ptimer("GatherRows(GPU)",r,x,((long long)g.n_ops())*x.n1);
	CUDA_STREAM(gatherRows_cu(r,x,gmaps,stream));
      }
    }


    template<typename TYPE>
    void weighted(const TensorView<TYPE>& _r, const TensorView<TYPE>& _x, const WeightedGatherMapB& g){
      auto r=_r.view2();
      r.n0*=g.out_columns;
      r.n1/=g.out_columns;
      r.s0/=g.out_columns;
      auto x=_x.view2();
      x.n0*=g.in_columns;
      x.n1/=g.in_columns;
      x.s0/=g.in_columns;

      if(_r.get_dev()==0){
	fnlog timer("GatherRows::weighted()");
	//logged_timer ptimer("GatherRows::weighted(CPU)",r,x,((long long)g.n_ops())*x.n1);
	CNINE_ASSRT(g.get_dev()==0);
	int N=g.size();
	for(int i=0; i<N; i++){
	  auto targt=r.slice0(g.target(i));
	  targt.n0=x.n1; // hack
	  int M=g.size_of(i);
	  for(int j=0; j<M; j++)
	    targt.add(x.slice0(g.src(i,j)),g.weight(i,j));
	}
      }

      if(_r.get_dev()==1){
	g.sort();
	fnlog timer("GatherRows::weighted()(G)");
	//logged_timer ptimer("GatherRows::weighted(GPU)",r,x,((long long)g.n_ops())*x.n1);
	//CUDA_STREAM(gatherRowsw_cu(r,x,g,stream));
      }
    }



  template<typename TYPE>
  void operator()(const TensorView<TYPE>& _r, const TensorView<TYPE>& _x, const FixedkGatherMap& g){
    CNINE_ASSRT(_r.ndims()==2);
    CNINE_ASSRT(_r.dim(0)%g.out_columns==0);
    CNINE_ASSRT(_x.ndims()==2);
    CNINE_ASSRT(_x.dim(0)%g.in_columns==0);

    auto r=_r.view2();
    r.n0/=g.out_columns;
    r.n1*=g.out_columns;
    auto x=_x.view2();
    x.n0/=g.in_columns;
    x.n1*=g.in_columns;
    
    if(_r.get_dev()==0){
      CNINE_ASSRT(g.get_dev()==0);
      int N=g.getn();
      int K=g.getk();
      for(int i=0; i<N; i++){
	int targt=g.target(i);
	for(int j=0; j<K; j++)
	  r.slice0(targt)+=x.slice0(g(i,j));
      }
    }

    if(_r.get_dev()==1){
      CUDA_STREAM(gatherRows_cu(r,x,g,stream));
    }
  }


  template<typename TYPE>
  Ltensor<TYPE> operator()(const Ltensor<TYPE>& x, const GatherMapB& g){
    CNINE_ASSRT(x.ndims()==2);
    Ltensor<TYPE> r({g.get_nout(),x.dim(1)},0,x.get_dev()); // take into account nc
    (*this)(r,x,g);
    return r;
  }

  template<typename TYPE>
  Ltensor<TYPE> operator()(const Ltensor<TYPE>& x, const FixedkGatherMap& g){
    CNINE_ASSRT(x.ndims()==2);
    Ltensor<TYPE> r({g.getn(),x.dim(1)},0,x.get_dev());
    (*this)(r,x,g);
    return r;
  }

  template<typename TYPE>
  Ltensor<TYPE> operator()(const Ltensor<TYPE>& x, const GatherMapPack& g){
    CNINE_ASSRT(x.ndims()==2);
    Ltensor<TYPE> r({g.get_nout(),x.dim(1)},0,x.get_dev());
    (*this)(r,x,g);
    return r;
  }

  };


  class GatherRowsMulti{
  public:

    template<typename TYPE>
    void operator()(TensorView<TYPE>& _r, const TensorView<TYPE>& _x, 
      const vector<shared_ptr<const GatherMapB> >& maps, const Ltensor<int>& out_offsets, const Ltensor<int>& in_offsets){
      FNTRACE();
      CNINE_ASSRT(maps.size()>0);
      const GatherMapB& g=*maps[0];
      CNINE_ASSRT(_r.ndims()==2);
      CNINE_ASSRT(_r.dim(1)%g.out_columns==0);
      CNINE_ASSRT(_x.ndims()==2);
      CNINE_ASSRT(_x.dim(1)%g.in_columns==0);
      CNINE_ASSRT(_r.get_dev()==1);

      auto r=_r.view2();
      r.n0*=g.out_columns;
      r.n1=r.n1*g.out_columns_n/g.out_columns;
      r.s0/=g.out_columns;
      auto x=_x.view2();
      x.n0*=g.in_columns;
      x.n1=x.n1*g.in_columns_n/g.in_columns;
      x.s0/=g.in_columns;
      CNINE_ASSRT(r.n1==x.n1);
    
      //CUDA_SAFE(cudaDeviceSynchronize());

      //g.sort();
      fnlog timer("GatherRowsMulti::operator()(G)");
      //logged_timer ptimer("GatherRows(GPU)",r,x,((long long)g.n_ops())*x.n1);
      CUDA_STREAM(gatherRowsMulti_cu(r,x,maps,out_offsets,in_offsets,stream));

    }
  };


  class MultiGatherRows{
  public:
    
    template<typename PACK>
    void operator()(PACK& _r, const PACK& _x, const GatherMapB& g){
    }    

  };

}

#endif 
