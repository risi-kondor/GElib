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


#ifndef _CnineLtensorEinsum2params
#define _CnineLtensorEinsum2params

#include "Ltensor.hpp"


namespace cnine{

  class Einsum2params{
  public:

    int tdims[4]={1,1,1,1}; // transfer
    int tstride_x[4]={0,0,0,0};
    int tstride_y[4]={0,0,0,0};
    int tstride_r[4]={0,0,0,0};

    int xsdims[4]={1,1,1,1};
    int xsstride[4]={0,0,0,0};

    int ysdims[4]={1,1,1,1};
    int ysstride[4]={0,0,0,0};

    int cdims[4]={1,1,1,1};
    int cstride_x[4]={0,0,0,0};
    int cstride_y[4]={0,0,0,0};

    int xdims[4]={1,1,1,1};
    int xstride_x[4]={0,0,0,0};
    int xstride_y[4]={0,0,0,0};
    int xstride_r[4]={0,0,0,0};

    int bdims[4]={1,1,1,1};
    int bstride[4]={0,0,0,0};

    int gstride_x[4]={0,0,0,0};
    int gstride_y[4]={0,0,0,0};
    int gstride_r[4]={0,0,0,0};

    bool convo_limiter[4]={false,false,false,false};

  public:

    template<typename TYPE>
    void sum1(const vector<vector<int> >& list, const TensorView<TYPE>& x){
      for(int i=0; i<list.size(); i++){
	xsdims[i]=x.dims[list[i][0]];
	xsstride[i]=x.strides.combine(list[i]);
      }
    }

    template<typename TYPE>
    void sum2(const vector<vector<int> >& list, const TensorView<TYPE>& x){
      for(int i=0; i<list.size(); i++){
	ysdims[i]=x.dims[list[i][0]];
	ysstride[i]=x.strides.combine(list[i]);
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
    void transfer2(int& ntransf, const vector<vector<int> >& yix, const vector<vector<int> >& rix, 
      const TensorView<TYPE>& y, const TensorView<TYPE>& r){
      for(int i=0; i<yix.size(); i++){
	tdims[ntransf]=r.dims[rix[i][0]];
	tstride_y[ntransf]=y.strides.combine(yix[i]);
	tstride_r[ntransf]=r.strides.combine(rix[i]);
	ntransf++;
      }
    }

    template<typename TYPE>
    void transfer12(int& ntransf, const vector<vector<int> >& xix, const vector<vector<int> >& yix, const vector<vector<int> >& rix, 
      const TensorView<TYPE>& x, const TensorView<TYPE>& y, const TensorView<TYPE>& r){
      for(int i=0; i<xix.size(); i++){
	tdims[ntransf]=x.dims[xix[i][0]];
	tstride_x[ntransf]=x.strides.combine(xix[i]);
	tstride_y[ntransf]=y.strides.combine(yix[i]);
	tstride_r[ntransf]=r.strides.combine(rix[i]);
	ntransf++;
      }
    }

    template<typename TYPE>
    void contract(int& ncontr, const vector<vector<int> >& xix, const vector<vector<int> >& yix, 
      const TensorView<TYPE>& x, const TensorView<TYPE>& y){
      for(int i=0; i<xix.size(); i++){
	cdims[ncontr]=std::min(x.dims[xix[i][0]],y.dims[yix[i][0]]); // min for the sake of convolution
	cstride_x[ncontr]=x.strides.combine(xix[i]);
	cstride_y[ncontr]=y.strides.combine(yix[i]);
	ncontr++;
      }
    }

    template<typename TYPE>
    void convolve_back(int& ncontr, const vector<vector<int> >& xix, const vector<vector<int> >& yix, 
      const TensorView<TYPE>& x, const TensorView<TYPE>& y){
      for(int i=0; i<xix.size(); i++){
	cdims[ncontr]=std::min(x.dims[xix[i][0]],y.dims[yix[i][0]]); // min for the sake of convolution
	cstride_x[ncontr]=-x.strides.combine(xix[i]); // note the - sign 
	cstride_y[ncontr]=y.strides.combine(yix[i]);
	convo_limiter[ncontr]=true;
	ncontr++;
      }
    }

    template<typename TYPE>
    void triple_contract(const vector<vector<vector<int> > >& list, 
      const TensorView<TYPE>& x, const TensorView<TYPE>& y, const TensorView<TYPE>& r){
     for(int i=0; i<list.size(); i++){
	xdims[i]=r.dims[list[i][2][0]];
	xstride_x[i]=x.strides.combine(list[i][0]);
	xstride_y[i]=y.strides.combine(list[i][1]);
	xstride_r[i]=r.strides.combine(list[i][2]);
      }
   }


  };

}

#endif 
