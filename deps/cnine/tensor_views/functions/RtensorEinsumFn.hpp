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


#ifndef _CnineRtensorEinsumFn
#define _CnineRtensorEinsumFn

#include "RtensorView.hpp"


namespace cnine{


  template<typename TYPE>
  class RtensorEinsumFn{
  public:

    vector<pair<vector<int>,vector<int> > > sstrides;
    vector<triple<vector<int> > > dstrides;
    vector<vector<int> > bstrides;

    RtensorEinsumFn(const string str){
      auto d0=str.find(",");
      auto d1=str.find("->");
      if(d0==string::npos || d1==string::npos || d0>d1){
	COUT("Error in RtensorEinsumFn: malformed einsum string");
	return;
      }
      auto xstr=str.substr(0,d0);
      auto ystr=str.substr(d0+1,d1-d0-1);
      auto rstr=str.substr(d1+2,string::npos);
      cout<<xstr<<endl;
      cout<<ystr<<endl;
      cout<<rstr<<endl;

      while(true){
	auto p=rstr.find_first_not_of('x');
	if(p==string::npos) break;
	char c=rstr[p];
	auto rindices=find_all(rstr,c);

	if(xstr.find(c)==string::npos && ystr.find(c)==string::npos){ // broadcast case
	  bstrides.push_back(rindices);
	}else{ // direct case
	  dstrides.push_back(triple<vector<int> >(rindices,find_all(xstr,c),find_all(ystr,c)));
	}
      }

      while(true){
	auto p=xstr.find_first_not_of('x');
	if(p==string::npos) break;
	char c=xstr[p];
	sstrides.push_back(pair<vector<int>,vector<int> >(find_all(xstr,c),find_all(ystr,c)));
      }
      
      if(false){
	cout<<"Direct:"<<endl;
	for(auto& p:dstrides){
	  cout<<p.second<<","<<p.third<<"->"<<p.first<<endl;
	}
	cout<<"Sum:"<<endl;
	for(auto& p:sstrides){
	  cout<<p.first<<","<<p.second<<endl;
	}
	cout<<"Broadcast:"<<endl;
	for(auto& p:bstrides){
	  cout<<p<<endl;
	}
      }

    }

      
  private:

    inline vector<int> find_all(string& str, const char c) const{
      vector<int> r;
      auto p=str.find_first_of(c);
      while(p!=string::npos){
	str.replace(p,1,1,'x');
	r.push_back(p);
	p=str.find_first_of(c);
      }
      return r;
    }


  public:


    void operator()(const RtensorView& r, const RtensorView& x, const RtensorView& y){
      
      if(dstrides.size()==0){
	sloops(r,x,y,0,0,0);
	return;
      }

      if(dstrides.size()==1){

	int I0=r.dims[dstrides[0].first[0]];
	int r0=r.strides.combine(dstrides[0].first);
	int x0=x.strides.combine(dstrides[0].second);
	int y0=y.strides.combine(dstrides[0].third);

	for(int i0=0; i0<I0; i0++)
	  sloops(r,x,y,i0*r0,i0*x0,i0*y0);
	return;
      }

      if(dstrides.size()==2){

	int I0=r.dims[dstrides[0].first[0]];
	int r0=r.strides.combine(dstrides[0].first);
	int x0=x.strides.combine(dstrides[0].second);
	int y0=y.strides.combine(dstrides[0].third);

	int I1=r.dims[dstrides[1].first[0]];
	int r1=r.strides.combine(dstrides[1].first);
	int x1=x.strides.combine(dstrides[1].second);
	int y1=y.strides.combine(dstrides[1].third);

	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    sloops(r,x,y,i0*r0+i1*r1,i0*x0+i1*x1,i0*y0+i1*y1);

	return;
      }

      if(dstrides.size()==3){

	int I0=r.dims[dstrides[0].first[0]];
	int r0=r.strides.combine(dstrides[0].first);
	int x0=x.strides.combine(dstrides[0].second);
	int y0=y.strides.combine(dstrides[0].third);

	int I1=r.dims[dstrides[1].first[0]];
	int r1=r.strides.combine(dstrides[1].first);
	int x1=x.strides.combine(dstrides[1].second);
	int y1=y.strides.combine(dstrides[1].third);

	int I2=r.dims[dstrides[2].first[0]];
	int r2=r.strides.combine(dstrides[2].first);
	int x2=x.strides.combine(dstrides[2].second);
	int y2=y.strides.combine(dstrides[2].third);

	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    for(int i2=0; i2<I2; i2++)
	    sloops(r,x,y,i0*r0+i1*r1+i2*r2,i0*x0+i1*x1+i2*x2,i0*y0+i1*y1+i2*y2);

	return;
      }
      
    }


  private:

    inline void sloops(const RtensorView& r, const RtensorView& x, const RtensorView& y, int ro, int xo, int yo){
      
      if(sstrides.size()==0){
	bloops(r,ro,x.arr[xo]*y.arr[yo]);
	return;
      }

      if(sstrides.size()==1){
	int I0=r.dims[sstrides[0].first[0]];
	int x0=r.strides.combine(sstrides[0].first);
	int y0=r.strides.combine(sstrides[0].second);

	TYPE t=0;
	for(int i0=0; i0<I0; i0++)
	  t+=x.arr[xo+i0*x0]*y.arr[yo+i0*y0];
	bloops(r,ro,t);
	return;
      }


      if(sstrides.size()==2){
	int I0=r.dims[sstrides[0].first[0]];
	int x0=r.strides.combine(sstrides[0].first);
	int y0=r.strides.combine(sstrides[0].second);

	int I1=r.dims[sstrides[1].first[0]];
	int x1=r.strides.combine(sstrides[1].first);
	int y1=r.strides.combine(sstrides[1].second);

	TYPE t=0;
	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    t+=x.arr[xo+i0*x0+i1*x1]*y.arr[yo+i0*y0+i1*y1];
	bloops(r,ro,t);
	return;
      }

      if(sstrides.size()==3){
	int I0=r.dims[sstrides[0].first[0]];
	int x0=r.strides.combine(sstrides[0].first);
	int y0=r.strides.combine(sstrides[0].second);

	int I1=r.dims[sstrides[1].first[0]];
	int x1=r.strides.combine(sstrides[1].first);
	int y1=r.strides.combine(sstrides[1].second);

	int I2=r.dims[sstrides[2].first[0]];
	int x2=r.strides.combine(sstrides[2].first);
	int y2=r.strides.combine(sstrides[2].second);

	TYPE t=0;
	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    for(int i2=0; i2<I2; i2++)
	      t+=x.arr[xo+i0*x0+i1*x1+i2*x2]*y.arr[yo+i0*y0+i1*y1+i2*y2];
	bloops(r,ro,t);
	return;

      }

    }


    inline void bloops(const RtensorView& r, int roffs, TYPE t){

      if(bstrides.size()==0){
	r.arr[roffs]+=t;
	return;
      }


      if(bstrides.size()==1){
	int I0=r.dims[bstrides[0][0]];
	int r0=r.strides.combine(bstrides[0]);

	for(int i0=0; i0<I0; i0++)
	  r.arr[roffs+i0*r0]+=t;
	return;
      }


      if(bstrides.size()==2){
	int I0=r.dims[bstrides[0][0]];
	int r0=r.strides.combine(bstrides[0]);
	int I1=r.dims[bstrides[1][0]];
	int r1=r.strides.combine(bstrides[1]);

	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    r.arr[roffs+i0*r0+i1*r1]+=t;
	return;
      }


      if(bstrides.size()==3){
	int I0=r.dims[bstrides[0][0]];
	int r0=r.strides.combine(bstrides[0]);
	int I1=r.dims[bstrides[1][0]];
	int r1=r.strides.combine(bstrides[1]);
	int I2=r.dims[bstrides[2][0]];
	int r2=r.strides.combine(bstrides[2]);

	for(int i0=0; i0<I0; i0++)
	  for(int i1=0; i1<I1; i1++)
	    for(int i2=0; i2<I2; i2++)
	      r.arr[roffs+i0*r0+i1*r1+i2*r2]+=t;
	return;
      }

    }


  };

}
 
#endif 
