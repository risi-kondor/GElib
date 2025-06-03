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


#ifndef _RtensorPackB
#define _RtensorPackB

#include "RtensorPack.hpp"


namespace cnine{


  class RtensorPackB: public RtensorPack{
  public:

    typedef RtensorA rtensor;
    int nc;


  public: // ---- Constructors -------------------------------------------------------------------------------


    RtensorPackB(){}

    RtensorPackB(const int ndims, const int _nc, const int _dev):
      RtensorPack(ndims,_dev), nc(_nc){}

    RtensorPackB(const IntTensor& _dir, const int _nc, const int _dev):
      RtensorPack(_dir,_dev), nc(_nc){}

    template<typename FILLTYPE>
    RtensorPackB(const int _N, const Gdims& _dims, const FILLTYPE& dummy, const int _dev=0):
      RtensorPack(_N,_dims,dummy,_dev), nc(_dims.back()){}

    template<typename FILLTYPE>
    RtensorPackB(const cnine::array_pool<int>& dims, const FILLTYPE& dummy, const int _dev=0):
      RtensorPack(dims,dummy,_dev){
      if(dims.size()>0) nc=dims(0).back();
      else nc=0;
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static RtensorPackB raw_like(const RtensorPackB& x){
      RtensorPackB R(x.dir,x.nc,x.dev);
      R.reserve(x.tail);
      R.tail=x.tail;
      return R;
    }

   static RtensorPackB zeros_like(const RtensorPackB& x){
      RtensorPackB R(x.dir,x.nc,x.dev);
      R.reserve(x.tail);
      R.tail=x.tail;
      if(x.dev==0) std::fill(R.arr,R.arr+x.tail,0);
      if(x.dev==1) CUDA_SAFE(cudaMemset(R.arrg,0,R.tail*sizeof(float)));
      return R;
    }

    static RtensorPackB zeros_like(const RtensorPackB& x, const int _nc){
      RtensorPackB R(x.dir,_nc,x.dev);
      int asize=x.tail*_nc/x.nc;
      R.reserve(asize);
      R.tail=asize;
      if(x.dev==0) std::fill(R.arr,R.arr+R.tail,0);
      if(x.dev==1) CUDA_SAFE(cudaMemset(R.arrg,0,R.tail*sizeof(float)));
      return R;
    }

    static RtensorPackB* new_zeros_like(const RtensorPackB& x){
      RtensorPackB*  R=new RtensorPackB(x.dir,x.nc,x.dev);
      R->reserve(x.tail);
      R->tail=x.tail;
      if(x.dev==0) std::fill(R->arr,R->arr+x.tail,0);
      if(x.dev==1) CUDA_SAFE(cudaMemset(R->arrg,0,R->tail*sizeof(float)));
      return R;
    }

    static RtensorPackB gaussian_like(const RtensorPackB& x){
      RtensorPackB R(x.dir,x.nc,0);
      R.reserve(x.tail);
      R.tail=x.tail;
      normal_distribution<double> distr;
      for(int i=0; i<x.tail; i++) R.arr[i]=distr(rndGen);
      return R.to_device(x.dev);
    }

    static RtensorPackB sequential_like(const RtensorPackB& x){
      RtensorPackB R(x.dir,x.nc,0);
      R.reserve(x.tail);
      R.tail=x.tail;
      normal_distribution<double> distr;
      for(int i=0; i<x.tail; i++) R.arr[i]=i;
      return R.to_device(x.dev);
    }

    static RtensorPackB cat(const vector<reference_wrapper<RtensorPackB> >& list){
      int _nc=0; if(list.size()>0) _nc=list[0].get().get_nc();
      return RtensorPackB(RtensorPack::cat
	(mapcar<reference_wrapper<RtensorPackB>,reference_wrapper<RtensorPack> >
	(list,[](const reference_wrapper<RtensorPackB>& x){
	  return reference_wrapper<RtensorPack>(x.get());})),_nc);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    RtensorPackB(const RtensorPackB& x):
      RtensorPack(x), nc(x.nc){}

    RtensorPackB(RtensorPackB&& x):
      RtensorPack(std::move(x)), nc(x.nc){}

    RtensorPackB& operator=(const RtensorPackB& x){
      RtensorPack::operator=(x);
      nc=x.nc;
      return *this;
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    RtensorPackB(const RtensorPack& x, const int _nc):
      RtensorPack(x), nc(_nc){}

    RtensorPackB(RtensorPack&& x, const int _nc):
      RtensorPack(std::move(x)), nc(_nc){}

    RtensorPackB(const rtensor& x){
      CNINE_ASSRT(x.ndims()==2);
      nc=x.dim(1);
      dev=x.dev;
      memsize=x.asize;
      tail=memsize;
      if(x.dev==0){
	arr=new float[std::max(memsize,1)];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
      dir=IntTensor({0,2},fill_noalloc());
      for(int i=0; i<x.dim(0); i++)
	dir.push_back({i*nc,nc});
    }

    RtensorPackB(rtensor&& x){
      if(x.is_view) {*this=RtensorPackB(x);return;}
      CNINE_ASSRT(x.ndims()==2);
      nc=x.dim(1);
      dev=x.dev;
      memsize=x.asize;
      tail=memsize;
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      dir=IntTensor({0,2},fill_noalloc());
      for(int i=0; i<x.dim(0); i++)
	dir.push_back({i*nc,nc});
    }

    RtensorPackB(const rtensor& x, const array_pool<int>& dims):
      RtensorPack(x,dims){
      nc=x.dim(1);
    }

    RtensorPackB(rtensor&& x, const array_pool<int>& dims):
      RtensorPack(std::move(x),dims){
      nc=x.dim(1);
    }

    rtensor tensor() const{
      return rtensor({tail/nc,nc},get_arr(),dev);
    }

    rtensor view_as_matrix() const{
      return rtensor::view_of_blob({tail/nc,nc},get_arr(),dev);
    }

    Rtensor2_view matrix_view() const{
      return Rtensor2_view(get_arr(),tail/nc,nc,nc,1,dev);
    }

    #ifdef _WITH_ATEN
    RtensorPackB(const at::Tensor& T):
      RtensorPackB(rtensor(T)){
    }
    #endif 


  public: // ---- Transport ----------------------------------------------------------------------------------


    RtensorPackB(const RtensorPackB& x, const int _dev): 
      RtensorPack(x,_dev){
      nc=x.nc;
    }

    RtensorPackB& to_device(const int _dev){
      RtensorPack::to_device(_dev);
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nc() const{
      return nc;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_mprod(const RtensorPackB& x, const rtensor& y){
      PTENS_ASSRT(x.size()==size());
      view_as_matrix().add_mprod(x.view_as_matrix(),y);
      //matrix_view().add_mprod(x.matrix_view(),y.view2());
    }

    void add_mprod_back0(const RtensorPackB& g, const rtensor& y){
      view_as_matrix().add_Mprod_AT(g.view_as_matrix(),y);
    }

    void add_mprod_back1_to(rtensor& r, const RtensorPackB& x) const{
      r.add_Mprod_TA(x.view_as_matrix(),view_as_matrix());
    }

    void add_scale_channels(const RtensorPackB& x, const Rtensor1_view& y){
      CNINE_DEVICE_SAME(x);
      CNINE_DEVICE_SAME(y);
      CNINE_ASSRT(x.tail==tail);
      CNINE_ASSRT(x.nc==nc);
      CNINE_ASSRT(y.n0==nc);
      if(dev==0){
	int n=tail/nc;
	for(int i=0; i<n; i++)
	  for(int j=0; j<nc; j++)
	    arr[i*nc+j]+=x.arr[i*nc+j]*y.arr[j];
      }
      if(dev==1){
	add(x.scale_channels(y));
	//CUBLAS_SAFE(cublasSdgmm(cnine_cublas,CUBLAS_SIDE_LEFT,nc,tail/nc,arrg,nc,y.arr,1,R.arrg,R.nc));
      }    
    }

    void add_bias(const rtensor& b){
      matrix_view().add_broadcast0(b.view1());
    }

    void add_bias_back1_to(const rtensor& b){
      matrix_view().sum0_into(b.view1());
    }

    void add_linear(const RtensorPackB& x, const rtensor& y, const rtensor& b){
      add_mprod(x,y);
      add_bias(b);
    }

    void add_linear_back0(const RtensorPackB& g, const rtensor& y){
      add_mprod_back0(g,y);
    }

    void add_linear_back1_to(rtensor& r, const RtensorPackB& x) const{
      add_mprod_back1_to(r,x);
    }

    void add_linear_back2_to(rtensor& b){
      add_bias_back1_to(b);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    rtensor inv_channel_norms() const{
      rtensor R(Gdims(nc),0);//,fill_zero());
      if(dev==0){
	int n=tail/nc;
	for(int i=0; i<nc; i++){
	  float t=0;
	  for(int j=0; j<n; j++){
	    float u=arr[j*nc+i];
	    t+=u*u;
	  }
	  R.set(i,1.0/sqrt(t));
	}
      }
      return R;
    }

    RtensorPackB scale_channels(const Rtensor1_view& y) const{
      RtensorPackB R=RtensorPackB::raw_like(*this);
      CNINE_DEVICE_SAME(y);
      CNINE_ASSRT(y.n0==nc);
      if(dev==0){
	int n=tail/nc;
	for(int i=0; i<n; i++)
	  for(int j=0; j<nc; j++)
	    R.arr[i*nc+j]=arr[i*nc+j]*y.arr[j];
      }
      if(dev==1){
	CUBLAS_SAFE(cublasSdgmm(cnine_cublas,CUBLAS_SIDE_LEFT,nc,tail/nc,arrg,nc,y.arr,1,R.arrg,R.nc));
      }
      return R;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "RtensorPackB";
    }

    friend ostream& operator<<(ostream& stream, const RtensorPackB& v){
      stream<<v.str(); return stream;}


  };

}


#endif 
    /*
    RtensorPackB(const rtensor& x):
      RtensorPack(x){
      nc=x.get_dim(1);
    }

    RtensorPackB(rtensor&& x):
      RtensorPack(x){
      nc=x.get_dim(1);
    }
    */
