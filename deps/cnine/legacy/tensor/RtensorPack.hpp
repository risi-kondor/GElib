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

#ifndef _RtensorPack
#define _RtensorPack

#include "array_pool.hpp"
#include "RtensorA.hpp"
#include "Rtensor1_view.hpp"
#include "Rtensor2_view.hpp"
#include "Rtensor3_view.hpp"
#include "IntTensor.hpp"


namespace cnine{

  class RtensorPack;

  #ifdef _WITH_CUDA
  extern void RtensorPack_add_ReLU_cu(RtensorPack& r, const RtensorPack& x, const float alpha, const cudaStream_t& stream);
  extern void RtensorPack_add_ReLU_back_cu(RtensorPack& r, const RtensorPack& g, const RtensorPack& x, const float alpha, const cudaStream_t& stream);
  #endif 

  class RtensorPack{
  public:

    typedef RtensorA rtensor;

    float* arr=nullptr;
    float* arrg=nullptr;
    int dev=0;
    int memsize=0;
    int tail=0;
    IntTensor dir;


    ~RtensorPack(){
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    RtensorPack(){}

    RtensorPack(const int ndims, const int _dev):
      dev(_dev), dir(Gdims(0,ndims+1),cnine::fill_noalloc()){}

    RtensorPack(const IntTensor& _dir, const int _dev):
      dev(_dev), dir(_dir){}

    RtensorPack(const int _N, const Gdims& _dims, const cnine::fill_raw& dummy, const int _dev=0):
      RtensorPack(_dims.size(),_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      for(int i=0; i<_N; i++)
	dir.push_back(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPack(const int _N, const Gdims& _dims, const cnine::fill_zero& dummy, const int _dev=0):
      RtensorPack(_dims.size(),_dev){
      int asize=_dims.asize();
      reserve(_N*asize);
      if(dev==0) std::fill(arr,arr+memsize,0);
      if(dev==1){CUDA_SAFE(cudaMemset(arrg,0,memsize*sizeof(float)))};
      for(int i=0; i<_N; i++)
	dir.push_back(i*asize,_dims);
      tail=_N*asize;
    }

    RtensorPack(const int _N, const Gdims& _dims, const cnine::fill_gaussian& dummy, const int _dev=0):
      RtensorPack(_dims.size(),0){
      int asize=_dims.asize();
      reserve(_N*asize);
      normal_distribution<double> distr;
      for(int i=0; i<_N*asize; i++) arr[i]=distr(rndGen)*dummy.c;
      for(int i=0; i<_N; i++){
	dir.push_back(i*asize,_dims);
      }
      tail=_N*asize;
      to_device(_dev);
    }

    RtensorPack(const int _N, const Gdims& _dims, const cnine::fill_sequential& dummy, const int _dev=0):
      RtensorPack(_dims.size(),0){
      int asize=_dims.asize();
      reserve(_N*asize);
      for(int i=0; i<_N*asize; i++) arr[i]=i;
      for(int i=0; i<_N; i++){
	dir.push_back(i*asize,_dims);
      }
      tail=_N*asize;
      to_device(_dev);
    }

    RtensorPack(const cnine::array_pool<int>& dimensions, const cnine::fill_zero& dummy, const int _dev=0):
      dev(_dev){
      if(dimensions.size()==0){
	reserve(0);
	dir=IntTensor(Gdims({0,0}));
	return;
      }

      dir=IntTensor(Gdims(0,dimensions(0).size()+1),cnine::fill_noalloc());

      int reserve_size=0;
      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	reserve_size+=t;
      }
      reserve(reserve_size);
      if(dev==0) std::fill(arr,arr+reserve_size,0);
      if(dev==1){CUDA_SAFE(cudaMemset(arrg,0,reserve_size*sizeof(float)))};

      for(int i=0; i<dimensions.size(); i++){
	vector<int> v=dimensions(i);
	int t=1; for(int j=0; j<v.size(); j++) t*=v[j];
	dir.push_back(tail,v);
	tail+=t;
      }
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static RtensorPack zeros_like(const RtensorPack& x){
      RtensorPack R(x.dir,x.dev);
      R.reserve(x.tail);
      R.tail=x.tail;
      if(x.dev==0) std::fill(R.arr,R.arr+x.tail,0);
      if(x.dev==1) CUDA_SAFE(cudaMemset(R.arrg,0,R.tail*sizeof(float)));
      return R;
    }

   static RtensorPack* new_zeros_like(const RtensorPack& x){
      RtensorPack*  R=new RtensorPack(x.dir,x.dev);
      R->reserve(x.tail);
      R->tail=x.tail;
      if(x.dev==0) std::fill(R->arr,R->arr+x.tail,0);
      if(x.dev==1) CUDA_SAFE(cudaMemset(R->arrg,0,R->tail*sizeof(float)));
      return R;
    }

     static RtensorPack gaussian_like(const RtensorPack& x){
      RtensorPack R(x.dir,0);
      R.reserve(x.tail);
      R.tail=x.tail;
      normal_distribution<double> distr;
      for(int i=0; i<x.tail; i++) R.arr[i]=distr(rndGen);
      return R.to_device(x.dev);
    }

    static RtensorPack cat(const vector<reference_wrapper<RtensorPack> >& list){
      int _dev=0;
      if(list.size()>0) _dev=list[0].get().dev;
      int t=0;
      for(auto& p:list) t+=p.get().tail;

      vector<reference_wrapper<IntTensor> > v; //(list.size());
      for(auto& p:list)
	v.push_back(p.get().dir);

      RtensorPack R(IntTensor::cat(v),_dev);
      int offs=0;
      int a=0;
      for(auto& _p:list){
	auto& p=_p.get();
	for(int i=0; i<p.size(); i++)
	  R.dir.set(a+i,0,R.dir(a+i,0)+offs);
	a+=p.size();
	offs+=p.tail;
      }

      R.reserve(t);
      for(auto& _p:list){
	auto& p=_p.get();
	CNINE_ASSRT(p.dev==_dev);
	if(_dev==0) std::copy(p.arr,p.arr+p.tail,R.arr+R.tail);
	if(_dev==1) CUDA_SAFE(cudaMemcpy(R.arrg+R.tail,p.arrg,p.tail*sizeof(float),cudaMemcpyDeviceToDevice));
	R.tail+=p.tail;
      }

      return R;
    }


  public: // ---- Memory management --------------------------------------------------------------------------


    void reserve(int n){
      if(n<=memsize && n>0) return;
      int newsize=n;
      if(dev==0){
	float* newarr=new float[std::max(newsize,1)];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	memsize=newsize;
      }
      if(dev==1){
	float* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, std::max(newsize,1)*sizeof(float)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	memsize=newsize;
      }
    }


    void reserve_zero(int n){
      if(n<=memsize) return;
      if(dev==0){
	float* newarr=new float[std::max(n,1)];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	std::fill(arr+memsize,arr+n,0);
	memsize=n;
      }
      if(dev==1){
	float* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, std::max(n,1)*sizeof(float)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	CUDA_SAFE(cudaMemset(arrg+memsize,0,(n-memsize)*sizeof(float)));
	memsize=n;
      }
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    RtensorPack(const RtensorPack& x):
      dev(x.dev),
      dir(x.dir){
      CNINE_COPY_WARNING();
      tail=x.tail;
      memsize=tail;
      if(dev==0){
	arr=new float[std::max(memsize,1)];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
    }

    RtensorPack(RtensorPack&& x):
      dev(x.dev),
      dir(std::move(x.dir)){
      CNINE_MOVE_WARNING();
      tail=x.tail; x.tail=0;
      memsize=x.memsize; x.memsize=0; 
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
    }

    RtensorPack& operator=(const RtensorPack& x){
      CNINE_ASSIGN_WARNING();
      dev=x.dev;
      dir=x.dir;
      tail=x.tail;
      memsize=tail;
      if(memsize==0) memsize=1;
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
      if(dev==0){
	arr=new float[std::max(memsize,1)];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(float)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice));  
      }
      return *this;
    }


  public: // ---- Conversions ---------------------------------------------------------------------------------


    RtensorPack(const rtensor& x, const cnine::array_pool<int>& dims){
      //CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(dims.size()>0);
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
      //int m=x.dim(1);
      dir=IntTensor({0,dims.size_of(0)+1},fill_noalloc());
      int t=0;
      for(int i=0; i<dims.size(); i++){
	Gdims D(dims(i));
	dir.push_back(t,D);
	t+=D.asize();
      }
      CNINE_ASSRT(t==tail);
    }

    RtensorPack(rtensor&& x, const cnine::array_pool<int>& dims){
      if(x.is_view) {*this=RtensorPack(x,dims);return;}
      //CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(dims.size()>0);
      dev=x.dev;
      memsize=x.asize;
      tail=memsize;
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      //int m=x.dim(1);
      dir=IntTensor({0,dims.size_of(0)+1,0},fill_noalloc());
      int t=0;
      for(int i=0; i<dims.size(); i++){
	Gdims D(dims(i));
	dir.push_back(t,D);
	t+=D.asize();
      }
      CNINE_ASSRT(t==tail);
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    RtensorPack(const RtensorPack& x, const int _dev): 
      dir(x.dir){
      dev=_dev;
      tail=x.tail;
      memsize=x.tail;
      if(dev==0){
	//cout<<"Copying RtensorPack to host"<<endl;
	arr=new float[std::max(memsize,1)];
	if(x.dev==0) std::copy(x.arr,x.arr+tail,arr);
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost));  
      }
      if(dev==1){
	//cout<<"Copying RtensorPack to device"<<endl;
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(float)));
	if(x.dev==0) CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(float),cudaMemcpyHostToDevice)); 
	if(x.dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(float),cudaMemcpyDeviceToDevice)); 
      }
    }


    RtensorPack& to_device(const int _dev){
      if(dev==_dev) return *this;

      if(_dev==0){
	if(dev==1){
	  //cout<<"Moving RtensorPack to host "<<tail<<endl;
	  memsize=tail;
	  delete[] arr;
	  arr=new float[std::max(memsize,1)];
	  CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(float),cudaMemcpyDeviceToHost));  
	  CUDA_SAFE(cudaFree(arrg));
	  arrg=nullptr;
	  dev=0;
	}
      }

      if(_dev>0){
	if(dev==0){
	  //cout<<"Moving RtensorPack to device "<<tail<<endl;
	  memsize=tail;
	  if(arrg) CUDA_SAFE(cudaFree(arrg));
	  CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(float)));
	  CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(float),cudaMemcpyHostToDevice));  
	  delete[] arr;
	  arr=nullptr;
	  dev=_dev;
	}
      }
      
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_dev() const{
      return dev;
    }

    int size() const{
      return dir.dim(0);
    }

    float* get_arr() const{
      if(dev==0) return arr;
      else return arrg;
    }


    int addr_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      return dir(i,0);
    }

    cnine::Gdims dims_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      return dir.row(i,1);
    }

    int dim_of(const int i, const int j) const{
      CNINE_IN_RANGE(i,size());
      return dir(i,1+j);
    }

    float* arr_of(const int i) const{
      if(dev==1) return arrg+addr_of(i);
      return arr+addr_of(i);
    }


    rtensor operator()(const int i) const{
      CNINE_IN_RANGE(i,size());
      return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    }

    rtensor view_of_tensor(const int i){
      CNINE_IN_RANGE(i,size());
      return rtensor::view_of_blob(dims_of(i),get_arr()+addr_of(i),dev);
    }

    const rtensor view_of_tensor(const int i) const{
      CNINE_IN_RANGE(i,size());
      return rtensor::view_of_blob(dims_of(i),get_arr()+addr_of(i),dev);
    }

    //rtensor tensor(const int i) const{
    //assert(i<size());
    //return rtensor(dims_of(i),get_arr()+addr_of(i),dev);
    //}

    Rtensor1_view view1_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      vector<int> v=dir.row(i);
      CNINE_ASSRT(v.size()==2);
      if(dev==1) return Rtensor1_view(arrg+v[0],v[1],1,1);
      return Rtensor1_view(arr+v[0],v[1],1,0);
    }

    Rtensor2_view view2_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      vector<int> v=dir.row(i);
      CNINE_ASSRT(v.size()==3);
      if(dev==1) return Rtensor2_view(arrg+v[0],v[1],v[2],v[2],1,1);
      return Rtensor2_view(arr+v[0],v[1],v[2],v[2],1,0);
    }

    Rtensor3_view view3_of(const int i) const{
      CNINE_IN_RANGE(i,size());
      vector<int> v=dir.row(i);
      CNINE_ASSRT(v.size()==4);
      if(dev==1) return Rtensor3_view(arrg+v[0],v[1],v[2],v[3],v[2]*v[3],v[3],1,1);
      return Rtensor3_view(arr+v[0],v[1],v[2],v[3],v[2]*v[3],v[3],1,0);
    }


    vector<int> headers(const int i) const{ // legacy
      return dir.row(i);
    }


    bool operator==(const RtensorPack& y) const{
      if(!(dir==y.dir)) return false;
      if(tail!=y.tail) return false;
      for(int i=0; i<tail; i++)
	if(arr[i]!=y.arr[i]) return false;
      return true;
    }

    //IntTensor* get_dirg_ptr(const int _dev=1) const{
    //if(!dirg) dirg=new IntTensor(dir,_dev);
    //return dirg;
    //}

    //int* dir_on_gpu(const int _dev=1) const{
    //if(!dirg) dirg=new IntTensor(dir,_dev);
    //return dirg->arrg;
    //}


  public: // ---- Push back ----------------------------------------------------------------------------------


    void push_back(const rtensor& x){
      CNINE_DEVICE_SAME(x);
      if(tail+x.asize>memsize)
	reserve(std::max(2*memsize,tail+x.asize));
      if(dev==0){
	std::copy(x.arr,x.arr+x.asize,arr+tail);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemcpy(arrg+tail,x.arrg,x.asize*sizeof(float),cudaMemcpyDeviceToDevice));
      }
      dir.push_back(tail,x.dims);
      tail+=x.asize;
    }

    void push_back_raw(const Gdims& _dims){
      int asize=_dims.asize();
      if(tail+asize>memsize)
	reserve(std::max(2*memsize,tail+asize));
      dir.push_back(tail,_dims);
      tail+=asize;
    }
      
    void push_back_zero(const Gdims& _dims){
      int asize=_dims.asize();
      if(tail+asize>memsize)
	reserve(std::max(2*memsize,tail+asize));
      if(dev==0){
	std::fill(arr+tail,arr+tail+asize,0);
      }
      if(dev==1){
	CUDA_SAFE(cudaMemset(arrg+tail,0,asize*sizeof(float)));
      }
      dir.push_back(tail,_dims);
      tail+=asize;
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const RtensorPack& x){
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(cnine::stdadd(x.arr,x.arr+tail,arr));
      GPUCODE(const float alpha = 1.0; CUBLAS_SAFE(cublasSaxpy(cnine_cublas, tail, &alpha, x.arrg, 1, arrg, 1)));
    }

    void add_subpack(const RtensorPack& x, const int offset){ // TODO dims checking
      CNINE_ASSRT(x.dev==dev);
      CNINE_ASSRT(offset<=x.size());
      if(offset==x.size()) return;
      int offs=x.dir(offset,0);
      CNINE_ASSRT(offs+tail<=x.tail);
      CPUCODE(cnine::stdadd(x.arr+offs,x.arr+offs+tail,arr));
      GPUCODE(const float alpha = 1.0; CUBLAS_SAFE(cublasSaxpy(cnine_cublas, tail, &alpha, x.arrg+offs, 1, arrg, 1)));
    }

    void add(const RtensorPack& x, const float c){
      assert(x.dev==dev);
      assert(x.tail==tail);
      CPUCODE(cnine::stdadd(x.arr,x.arr+tail,arr,c));
      GPUCODE(const float alpha = c; CUBLAS_SAFE(cublasSaxpy(cnine_cublas, tail, &alpha, x.arrg, 1, arrg, 1)));
    }


    void add_ReLU(const RtensorPack& x, const float alpha=0.1){
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.tail==tail);
      CPUCODE(for(int i=0; i<tail; i++) if(x.arr[i]>0) arr[i]+=x.arr[i]; else arr[i]+=alpha*x.arr[i]);
      GPUCODE(CUDA_STREAM(RtensorPack_add_ReLU_cu(*this,x,alpha,stream)));
    }

    void add_ReLU_back(const RtensorPack& g, const RtensorPack& x, const float alpha=0.1){
      CNINE_DEVICE_SAME(g);
      CNINE_DEVICE_SAME(x);
      CNINE_ASSRT(x.tail==tail);
      CPUCODE(for(int i=0; i<tail; i++) if(x.arr[i]>0) arr[i]+=g.arr[i]; else arr[i]+=g.arr[i]*alpha);
      GPUCODE(CUDA_STREAM(RtensorPack_add_ReLU_back_cu(*this,g,x,alpha,stream)));
    }


  public: // ---- Operations ---------------------------------------------------------------------------------

    
    float inp(const RtensorPack& y) const{
      CNINE_ASSRT(tail==y.tail);
      if(dev==0){
	if(y.dev>0) return inp(RtensorPack(y,0));
	float t=0;
	for(int i=0; i<tail; i++) t+=arr[i]*y.arr[i];
	return t;
      }
      if(dev==1){
	CNINE_ASSRT(y.dev==1);
	float r=0;
	CUBLAS_SAFE(cublasSdot(cnine_cublas,tail,arrg,1,y.arrg,1,&r));
	return r;
      }
      return 0;
    }

    float diff2(const RtensorPack& y) const{
      CNINE_CPUONLY();
      if(y.dev>0) return diff2(RtensorPack(y,0));
      CNINE_ASSRT(tail==y.tail);
      float t=0;
      CPUCODE(for(int i=0; i<tail; i++) t+=(arr[i]-y.arr[i])*(arr[i]-y.arr[i]);)
      return t;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "RtensorPack";
    }

    string repr() const{
      return "<RtensorPack[N="+to_string(size())+"]>";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<"Tensor "<<i<<":"<<endl;
	oss<<(*this)(i).str(indent)<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const RtensorPack& v){
      stream<<v.str(); return stream;}

  };


}

#endif
