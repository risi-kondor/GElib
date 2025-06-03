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


#ifndef _CnineEinsum1
#define _CnineEinsum1

#include "TensorView.hpp"
#include "EinsumForm1.hpp"
#include "Einsum1params.hpp"
#include "GatherMapB.hpp"


namespace cnine{

#ifdef _WITH_CUDA
  void add_einsum1_cu(const TensorView<float>& r, const TensorView<float>& x,  
    const Einsum1params& params, const cudaStream_t& stream);
#endif 


  class Einsum1{
  public:

    EinsumForm1 form;

    Einsum1(const string str):
      form(str){
    }

    template<typename TYPE>
    TensorView<TYPE> operator()(const TensorView<TYPE>& x, vector<int> rdims={}){
      CNINE_ASSRT(rdims.size()==form.bcast_ids.size());
      
      vector<int> dimensions(form.id_tail,-1);
      for(int i=0; i<form.x_ids.size(); i++){
	if(dimensions[form.x_ids[i]]==-1)
	  dimensions[form.x_ids[i]]=x.dims[i];
	else
	  CNINE_ASSRT(dimensions[form.x_ids[i]]==x.dims[i]);
      }

      for(int i=0; i<form.bcast_ids.size(); i++)
	dimensions[form.bcast_ids[i]]=rdims[i];

      auto r_dims=mapcar<int,int>(form.r_ids,[&](const int& id){return dimensions[id];});

      TensorView<TYPE> R(r_dims,0,x.get_dev());
      add_einsum(R,x);
      return R;
    }

    
    template<typename TYPE>
    TensorView<TYPE> operator()(const TensorView<TYPE>& x, const GatherMapB& gmap, vector<int> rdims={}){
      CNINE_ASSRT(rdims.size()==form.bcast_ids.size());
      
      vector<int> dimensions(form.id_tail,-1);
      for(int i=0; i<form.x_ids.size(); i++){
	if(dimensions[form.x_ids[i]]==-1)
	  dimensions[form.x_ids[i]]=x.dims[i];
	else
	  CNINE_ASSRT(dimensions[form.x_ids[i]]==x.dims[i]);
      }

      for(int i=0; i<form.bcast_ids.size(); i++)
	dimensions[form.bcast_ids[i]]=rdims[i];

      auto r_dims=mapcar<int,int>(form.r_ids,[&](const int& id){return dimensions[id];});

      for(auto p:form.xr_gather) // currently only one gather allowed
	for(auto j:p[2]) r_dims[j]=gmap.n_in;
      
      TensorView<TYPE> R(r_dims,0,x.get_dev());
      add_einsum(R,x,gmap);
      return R;
    }

    
    template<typename TYPE, typename TRANSFORM>
    TensorView<TYPE> operator()(const TensorView<TYPE>& x, const TRANSFORM& transf, vector<int> rdims={}){
      CNINE_ASSRT(rdims.size()==form.bcast_ids.size());
      
      vector<int> dimensions(form.id_tail,-1);
      for(int i=0; i<form.x_ids.size(); i++){
	if(dimensions[form.x_ids[i]]==-1)
	  dimensions[form.x_ids[i]]=x.dims[i];
	else
	  CNINE_ASSRT(dimensions[form.x_ids[i]]==x.dims[i]);
      }

      for(int i=0; i<form.bcast_ids.size(); i++)
	dimensions[form.bcast_ids[i]]=rdims[i];

      auto r_dims=mapcar<int,int>(form.r_ids,[&](const int& id){return dimensions[id];});

      TensorView<TYPE> R(r_dims,0,x.get_dev());
      add_einsum(R,x,transf);
      return R;
    }

    
  public: // ---- cumulative operations -----------------------------------------------------------------------------------------


    template<typename TYPE>
    void add_einsum(const TensorView<TYPE>& r, const TensorView<TYPE>& x){

      auto& x_summation_indices=form.x_summation_indices;
      auto& r_summation_indices=form.r_summation_indices;
      auto& xr_indices=form.xr_indices;
      auto& xr_gather=form.xr_gather;

      CNINE_ASSRT(x_summation_indices.size()<=3);
      CNINE_ASSRT(r_summation_indices.size()<=3);
      CNINE_ASSRT(xr_indices.size()<=4);
      CNINE_ASSRT(xr_gather.size()==0);

      Einsum1params params;

      params.sum1(x_summation_indices,x);
      params.bcast(r_summation_indices,r);

      int ntransf=0;
      params.transfer1(ntransf,xr_indices.vecs[0],xr_indices.vecs[1],x,r);

      add_einsum(r,x,params);
    }

      
    template<typename TYPE, typename ARG0>
    void add_einsum(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const ARG0& arg){

      auto& x_summation_indices=form.x_summation_indices;
      auto& r_summation_indices=form.r_summation_indices;
      auto& xr_indices=form.xr_indices;
      auto& xr_gather=form.xr_gather;

      CNINE_ASSRT(x_summation_indices.size()<=3);
      CNINE_ASSRT(r_summation_indices.size()<=3);
      CNINE_ASSRT(xr_indices.size()<=4);
      //CNINE_ASSRT(xr_gather.size()==1);

      Einsum1params params;

      params.sum1(x_summation_indices,x);
      params.bcast(r_summation_indices,r);

      int ntransf=0;
      params.transfer1(ntransf,xr_indices.vecs[0],xr_indices.vecs[1],x,r);

      params.gather1(xr_gather.vecs[0],xr_gather.vecs[1],x,r);

      add_einsum(r,x,arg,params);
    }

      
    template<typename TYPE>
    void add_einsum_back(const TensorView<TYPE>& x, const TensorView<TYPE>& r){

      auto& x_summation_indices=form.x_summation_indices;
      auto& r_summation_indices=form.r_summation_indices;
      auto& xr_indices=form.xr_indices;

      CNINE_ASSRT(x_summation_indices.size()<=3);
      CNINE_ASSRT(r_summation_indices.size()<=3);
      CNINE_ASSRT(xr_indices.size()<=4);

      Einsum1params params;

      params.sum1(r_summation_indices,r);
      params.bcast(x_summation_indices,x);

      int ntransf=0;
      params.transfer1(ntransf,xr_indices.vecs[1],xr_indices.vecs[0],r,x);

      add_einsum(x,r,params);
    }


  public: // ----------------------------------------------------------------------------------------------------------


    template<typename TYPE>
    void add_einsum(const TensorView<TYPE>& _r, const TensorView<TYPE>& x, const Einsum1params& p, 
      const int _xoffs=0, const int _roffs=0){
      auto& r=const_cast<TensorView<TYPE>& >(_r);

      if(_r.get_dev()==1){
	CUDA_STREAM(add_einsum1_cu(_r,x,p,stream));
	return;
      }

      int xoffs=_xoffs;
      int roffs=_roffs;

      for(int t0=0; t0<p.tdims[0]; t0++)
	for(int t1=0; t1<p.tdims[1]; t1++)
	  for(int t2=0; t2<p.tdims[2]; t2++)
	    for(int t3=0; t3<p.tdims[3]; t3++){
	      int xoffs_t=xoffs+t0*p.tstride_x[0]+t1*p.tstride_x[1]+t2*p.tstride_x[2]+t3*p.tstride_x[3];
	      int roffs_t=roffs+t0*p.tstride_r[0]+t1*p.tstride_r[1]+t2*p.tstride_r[2]+t3*p.tstride_x[3];

	      TYPE xt=0;
	      for(int xs0=0; xs0<p.xsdims[0]; xs0++)
		for(int xs1=0; xs1<p.xsdims[1]; xs1++)
		  for(int xs2=0; xs2<p.xsdims[2]; xs2++)
		    xt+=*(x.get_arr()+xoffs_t+xs0*p.xsstride[0]+xs1*p.xsstride[1]+xs2*p.xsstride[2]);

	      for(int b0=0; b0<p.bdims[0]; b0++)
		for(int b1=0; b1<p.bdims[1]; b1++)
		  for(int b2=0; b2<p.bdims[2]; b2++)
		    *(r.get_arr()+roffs_t+b0*p.bstride[0]+b1*p.bstride[1]+b2*p.bstride[2])+=xt;
	    }
    }
    

    template<typename TYPE>
    void add_einsum(const TensorView<TYPE>& _r, const TensorView<TYPE>& x, 
      const GatherMapB& gmap, const Einsum1params& p){
      auto& r=const_cast<TensorView<TYPE>& >(_r);


      if(_r.get_dev()==1){
	//CUDA_STREAM(add_einsum1_cu(_r,x,gmap,p,stream));
	return;
      }

      gmap.for_each([&](const int i, const int j){
	  //cout<<i<<j<<endl;
	  add_einsum(r,x,p,j*p.gstride_x[0],i*p.gstride_r[0]);});
    }    


    template<typename TYPE, typename TRANSFORM>
    void add_einsum(const TensorView<TYPE>& _r, const TensorView<TYPE>& x, const Einsum1params& p, 
      const TRANSFORM& transf){
      auto& r=const_cast<TensorView<TYPE>& >(_r);

      if(_r.get_dev()==1){
	//CUDA_STREAM(add_einsum1_cu(_r,x,p,stream));
	return;
      }

      int xoffs=0;
      int roffs=0;

      for(int t0=0; t0<p.tdims[0]; t0++)
	for(int t1=0; t1<p.tdims[1]; t1++)
	  for(int t2=0; t2<p.tdims[2]; t2++)
	    for(int t3=0; t3<p.tdims[3]; t3++){
	      int xoffs_t=xoffs+t0*p.tstride_x[0]+t1*p.tstride_x[1]+t2*p.tstride_x[2]+t3*p.tstride_x[3];
	      int roffs_t=roffs+t0*p.tstride_r[0]+t1*p.tstride_r[1]+t2*p.tstride_r[2]+t3*p.tstride_x[3];

	      TYPE xt=0;
	      for(int xs0=0; xs0<p.xsdims[0]; xs0++)
		for(int xs1=0; xs1<p.xsdims[1]; xs1++)
		  for(int xs2=0; xs2<p.xsdims[2]; xs2++)
		    xt+=*(x.get_arr()+xoffs_t+xs0*p.xsstride[0]+xs1*p.xsstride[1]+xs2*p.xsstride[2]);

	      for(int b0=0; b0<p.bdims[0]; b0++)
		for(int b1=0; b1<p.bdims[1]; b1++)
		  for(int b2=0; b2<p.bdims[2]; b2++)
		    *(r.get_arr()+roffs_t+b0*p.bstride[0]+b1*p.bstride[1]+b2*p.bstride[2])+=xt;
	    }
    }
    


  };

}

#endif 
