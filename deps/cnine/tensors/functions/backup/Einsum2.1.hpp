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


#ifndef _CnineEinsum2
#define _CnineEinsum2

#include "Ltensor.hpp"
#include "EinsumForm2.hpp"
#include "EinsumParams.hpp"


namespace cnine{

#ifdef _WITH_CUDA
  void add_einsum2_cu(const TensorView<float>& r, const TensorView<float>& x, const TensorView<float>& y, 
    const Einsum2params& params, const cudaStream_t& stream)
#endif 


  class Einsum2{
  public:

    EinsumForm2 form;

    Einsum2(const string str):
      form(str){}

    template<typename TYPE>
    TensorView<TYPE> operator()(const TensorView<TYPE>& x, const TensorView<TYPE>& y, vector<int> rdims={}){
      CNINE_ASSRT(rdims.size()==form.r_ids.size());
      
      vector<int> dimensions(form.id_tail,-1);
      for(int i=0; i<form.x_ids.size(); i++){
	if(dimensions[form.x_ids[i]]==-1)
	  dimensions[form.x_ids[i]]=x.dims[i];
	else
	  CNINE_ASSRT(dimensions[form.x_ids[i]]==x.dims[i]);
      }
      for(int i=0; i<form.y_ids.size(); i++){
	if(dimensions[form.y_ids[i]]==-1)
	  dimensions[form.y_ids[i]]=y.dims[i];
	else
	  CNINE_ASSRT(dimensions[form.y_ids[i]]==y.dims[i]);
      }

      for(int i=0; i<form.r_ids.size(); i++)
	dimensions[form.r_ids[i]]=rdims[i];

      auto r_dims=mapcar<int,int>(form.r_ids,[&](const int& id){return dimensions[id];});
      for(int i=0; i<form.transfer_indices.size(); i++){ // special treatment for convolutions
	if(form.convolution_flag[i]){
	  int d=x.dims[form.transfer_indices[i][0][0]]-x.dims[form.transfer_indices[i][1][0]]+1;
	  for(auto j:form.transfer_indices[i][2]) r_dims[j]=d;
	}
      }
	
      TensorView<TYPE> R(r_dims,0,x.get_dev());
      add_einsum(R,x,y);
      return R;
    }


    template<typename TYPE>
    void add_einsum(const TensorView<TYPE>& r, const TensorView<TYPE>& x, const TensorView<TYPE>& y){

      auto& x_summation_indices=form.x_summation_indices;
      auto& y_summation_indices=form.y_summation_indices;
      auto& r_summation_indices=form.r_summation_indices;
      auto& xr_indices=form.xr_indices;
      auto& yr_indices=form.yr_indices;
      auto& xy_indices=form.xy_indices;
      auto& transfer_indices=form.transfer_indices;
      auto& convolution_indices=form.convolution_indices;
      auto& triple_contraction_indices=form.triple_contraction_indices;
      //auto& convolution_flag=form.convolution_flag;

      //int nconvolutions=0;
      //for(auto p:convolution_flag)
      //nconvolutions+=p;

      CNINE_ASSRT(x_summation_indices.size()<=3);
      CNINE_ASSRT(y_summation_indices.size()<=3);
      CNINE_ASSRT(r_summation_indices.size()<=3);
      CNINE_ASSRT(transfer_indices.size()+xr_indices.size()+yr_indices.size()+convolution_indices.size()<=4);
      CNINE_ASSRT(xy_indices.size()+convolution_indices.size()<=3);
      CNINE_ASSRT(triple_contraction_indices.size()<=1);

      Einsum2params params;

      params.sum1(x_summation_indices,x);
      params.sum2(y_summation_indices,y);
      params.bcast(r_summation_indices,r);

      int ntransf=0;
      params.transfer1(ntransf,xr_indices,x,r);
      params.transfer2(ntransf,yr_indices,y,r);
      params.transfer12(ntransf,transfer_indices,x,y,r);
      params.transfer12(ntransf,convolution_indices,x,y,r);

      int ncontr=0;
      params.contract(ncontr,xy_indices,x,y);
      params.convolve(ncontr,convolution_indices,r,x,y);
     
      add_einsum(r,x,y,params);
    }


    template<typename TYPE>
    void add_einsum_back0(const TensorView<TYPE>& x, const TensorView<TYPE>& r, const TensorView<TYPE>& y){

      auto& x_summation_indices=form.x_summation_indices;
      auto& y_summation_indices=form.y_summation_indices;
      auto& r_summation_indices=form.r_summation_indices;
      auto& xr_indices=form.xr_indices;
      auto& yr_indices=form.yr_indices;
      auto& xy_indices=form.xy_indices;
      auto& transfer_indices=form.transfer_indices;
      auto& convolution_indices=form.convolution_indices;
      auto& triple_contraction_indices=form.triple_contraction_indices;
      auto& convolution_flag=form.convolution_flag;

      //int nconvolutions=0;
      //for(auto p:convolution_flag)
      //nconvolutions+=p;

      CNINE_ASSRT(x_summation_indices.size()<=3);
      CNINE_ASSRT(y_summation_indices.size()<=3);
      CNINE_ASSRT(r_summation_indices.size()<=3);
      CNINE_ASSRT(transfer_indices.size()+xr_indices.size()+yr_indices.size()<=4);
      CNINE_ASSRT(xy_indices.size()+nconvolutions<=3);
      CNINE_ASSRT(triple_contraction_indices.size()<=1);

      Einsum2params params;

      params.sum1(r_summation_indices,r);
      params.sum2(y_summation_indices,y);
      params.bcast(x_summation_indices,x);

      int ntransf=0;
      params.transfer1_back(ntransf,xr_indices,x,r);
      params.transfer2_back(ntransf,xy_indices,x,y);
      params.transfer12_back0(ntransf,transfer_indices,x,y,r);

      int ncontr=0;
      params.contract_back(ncontr,yr_indices,y,r);
      params.convolve_back0(ncontr,transfer_indices,convolution_flag,x,y,r);
     
      add_einsum(x,r,y,params);
    }


 

  public: // ----------------------------------------------------------------------------------------------------------


    template<typename TYPE>
    void add_einsum(const TensorView<TYPE>& _r, const TensorView<TYPE>& x, const TensorView<TYPE>& y, const Einsum2params& p){
      auto& r=const_cast<TensorView<TYPE>& >(_r);

      if(_r.get_dev()==1)
	CUDA_STREAM(add_einsum2_cu(_r,x,y,p,stream));

      int xoffs=0;
      int yoffs=0;
      int roffs=0;

      // transfer loops
      for(int t0=0; t0<p.tdims[0]; t0++)
	for(int t1=0; t1<p.tdims[1]; t1++)
	  for(int t2=0; t2<p.tdims[2]; t2++)
	    for(int t3=0; t3<p.tdims[3]; t3++){
	      int xoffs_t=xoffs+t0*p.tstride_x[0]+t1*p.tstride_x[1]+t2*p.tstride_x[2]+t3*p.tstride_x[2];
	      int yoffs_t=yoffs+t0*p.tstride_y[0]+t1*p.tstride_y[1]+t2*p.tstride_y[2]+t3*p.tstride_x[2];
	      int roffs_t=roffs+t0*p.tstride_r[0]+t1*p.tstride_r[1]+t2*p.tstride_r[2]+t3*p.tstride_x[2];

	      TYPE t=0;
	      // contraction loops
	      for(int c0=0; c0<p.cdims[0]; c0++)
		for(int c1=0; c1<p.cdims[1]; c1++)
		  for(int c2=0; c2<p.cdims[2]; c2++){
		    int xoffs_c=xoffs_t+c0*p.cstride_x[0]+c1*p.cstride_x[1]+c2*p.cstride_x[2];
		    int yoffs_c=yoffs_t+c0*p.cstride_y[0]+c1*p.cstride_y[1]+c2*p.cstride_y[2];

		    TYPE xt=0;
		    for(int xs0=0; xs0<p.xsdims[0]; xs0++)
		      for(int xs1=0; xs1<p.xsdims[1]; xs1++)
			for(int xs2=0; xs2<p.xsdims[2]; xs2++)
			  xt+=*(x.get_arr()+xoffs_c+xs0*p.xsstride[0]+xs1*p.xsstride[1]+xs2*p.xsstride[2]);

		    TYPE yt=0;
		    for(int ys0=0; ys0<p.ysdims[0]; ys0++)
		      for(int ys1=0; ys1<p.ysdims[1]; ys1++)
			for(int ys2=0; ys2<p.ysdims[2]; ys2++)
			  yt+=*(y.get_arr()+yoffs_c+ys0*p.ysstride[0]+ys1*p.ysstride[1]+ys2*p.ysstride[2]);

		    t+=xt*yt;
		  }
	      
	      // broadcast loops
	      for(int b0=0; b0<p.bdims[0]; b0++)
		for(int b1=0; b1<p.bdims[1]; b1++)
		  for(int b2=0; b2<p.bdims[2]; b2++)
		    *(r.get_arr()+roffs_t+b0*p.bstride[0]+b1*p.bstride[1]+b2*p.bstride[2])+=t;
	    }

    }
    

  };

}

#endif 
	      // convolution loops
	      //for(int u0=0; u0<p.udims[0]; u0++)
	      //for(int u1=0; u1<p.udims[1]; u1++)
	      //for(int u2=0; u2<p.udims[2]; u2++)
	      //for(int u3=0; u3<p.udims[3]; u3++){
	      //int xoffs_u=xoffs_u+u0*p.tstride_x[0]+u1*p.tstride_x[1]+u2*p.tstride_x[2]+u3*p.tstride_x[3];
	      //int yoffs_u=yoffs_u+u0*p.tstride_y[0]+u1*p.tstride_y[1]+u2*p.tstride_y[2]+u3*p.tstride_y[3];

		    
     /*
      for(auto& p:xr_indices){
	params.tdims[ntransf]=x.dims[p.first[0]];
	params.tstride_x[ntransf]=x.strides.combine(p.first);
	params.tstride_r[ntransf]=r.strides.combine(p.second);
	ntransf++;
      }
      for(auto& p:yr_indices){
	params.tdims[ntransf]=y.dims[p.first[0]];
	params.tstride_y[ntransf]=y.strides.combine(p.first);
	params.tstride_r[ntransf]=r.strides.combine(p.second);
	ntransf++;
      }
      for(auto& p:transfer_indices){
	params.tdims[ntransf]=x.dims[p[0][0]];
	params.tstride_x[ntransf]=x.strides.combine(p[0]);
	params.tstride_y[ntransf]=y.strides.combine(p[1]);
	params.tstride_r[ntransf]=r.strides.combine(p[2]);
	ntransf++;
      }
      */
     /*
      for(auto& p:xy_indices){
	params.cdims[ncontr]=x.dims[p.first[0]];
	params.cstride_x[ncontr]=x.strides.combine(p.first);
	params.cstride_y[ncontr]=y.strides.combine(p.second);
	ncontr++;
      }
      for(int i=0; i<transfer_indices.size(); i++){
	if(convolution_flag[i]){
	  params.cdims[ncontr]=y.dims[transfer_indices[i][1][0]];
	  params.cstride_x[ncontr]=x.strides.combine(transfer_indices[i][0]);
	  params.cstride_y[ncontr]=y.strides.combine(transfer_indices[i][1]);
	  ncontr++;
	}
       }
      */
      /*
      for(int i=0; i<triple_contraction_indices.size(); i++){
	params.xdims[i]=r.dims[triple_contraction_indices[i][2][0]];
	params.xstride_x[i]=x.strides.combine(triple_contraction_indices[i][0]);
	params.xstride_y[i]=y.strides.combine(triple_contraction_indices[i][1]);
	params.xstride_r[i]=r.strides.combine(triple_contraction_indices[i][2]);
      }
      */

