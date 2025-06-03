/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineLtensorEinsum1Params
#define _CnineLtensorEinsum1Params

#include "Cnine_base.hpp"


namespace cnine{


  struct Einsum1params{
  public:

    int tdims[4]={1,1,1,1};
    int tstride_x[4]={0,0,0,0};
    int tstride_r[4]={0,0,0,0};

    int xsdims[4]={1,1,1,1};
    int xsstride[4]={0,0,0,0};

    int bdims[4]={1,1,1,1};
    int bstride[4]={0,0,0,0};

    int gstride_x[4]={0,0,0,0};
    int gstride_r[4]={0,0,0,0};


  public:

    template<typename TYPE>
    void sum1(const vector<vector<int> >& list, const TensorView<TYPE>& x){
      for(int i=0; i<list.size(); i++){
	xsdims[i]=x.dims[list[i][0]];
	xsstride[i]=x.strides.combine(list[i]);
      }
    }

    template<typename TYPE>
    void bcast(const vector<vector<int> >& list, const TensorView<TYPE>& x){
      for(int i=0; i<list.size(); i++){
	bdims[i]=x.dims[list[i][0]];
	bstride[i]=x.strides.combine(list[i]);
      }
    }

    template<typename TYPE>
    void transfer1(int& ntransf, const vector<vector<int> >& xix, const vector<vector<int> >& rix, 
      const TensorView<TYPE>& x, const TensorView<TYPE>& r){
      for(int i=0; i<xix.size(); i++){
	tdims[ntransf]=r.dims[rix[i][0]];
	tstride_x[ntransf]=x.strides.combine(xix[i]);
	tstride_r[ntransf]=r.strides.combine(rix[i]);
	ntransf++;
      }
    }

    template<typename TYPE>
    void gather1(const vector<vector<int> >& xix, const vector<vector<int> >& rix, 
      const TensorView<TYPE>& x, const TensorView<TYPE>& r){
      for(int i=0; i<xix.size(); i++){
	gstride_x[i]=x.strides.combine(xix[i]);
	gstride_r[i]=r.strides.combine(rix[i]);
      }
    }



  };



}

#endif 
