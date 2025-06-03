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

#ifndef _array_pool
#define _array_pool

#include "Cnine_base.hpp"
#include "IntTensor.hpp"
#include "Itensor1_view.hpp"
#include "Rtensor1_view.hpp"
//#include "TensorView.hpp"
//#include "Tensor.hpp"
#include "Ltensor.hpp"


namespace cnine{

  template<typename TYPE>
  class array_pool;

  template<typename TYPE>
  class hlists;

  inline Itensor1_view view_of_part(const array_pool<int>&, const int);
  inline Rtensor1_view view_of_part(const array_pool<float>&, const int);
  class TensorPackDir;
  template<typename TYPE> class CSRmatrix;


  template<typename TYPE>
  class array_pool{
    //private:
  public:

    TYPE* arr=nullptr;
    TYPE* arrg=nullptr;
    int memsize=0;
    int tail=0;
    int dev=0;
    bool is_view=false;
    array_pool* gpu_clone=nullptr;

  public: 

    IntTensor dir; // should become private 


  public:

    friend class hlists<TYPE>;
    friend class TensorPackDir; // decomission this
    friend class CSRmatrix<TYPE>;
    friend Itensor1_view view_of_part(const array_pool<int>&, const int);
    friend Rtensor1_view view_of_part(const array_pool<float>&, const int);

    ~array_pool(){
      if(is_view) return;
      if(arr) delete[] arr;
      if(arrg) {CUDA_SAFE(cudaFree(arrg));}
      if(gpu_clone) delete gpu_clone;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    array_pool(): 
      dir(Gdims({0,2}),cnine::fill_noalloc()){}

    array_pool(const int n): 
      dir(Gdims({n,2})){}

    array_pool(const int n, const int m, const int _dev=0): 
      memsize(n*m),
      tail(n*m),
      dev(_dev),
      dir(Gdims({n,2})){
      for(int i=0; i<n; i++){
	dir.set(i,0,i*m);
	dir.set(i,1,m);
      }
      CPUCODE(arr=new TYPE[std::max(n*m,1)]);
      GPUCODE(CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(TYPE))));
    }

    array_pool(const int n, const int m, const fill_sequential& dummy, const int _dev=0): 
      array_pool(n,m){
      for(int i=0; i<n*m; i++)
	arr[i]=i;
      to_device(_dev);
    }

    array_pool(const int n, const int _total, const fill_reserve& dummy, const int _dev=0): 
      memsize(_total),
      tail(0),
      dev(_dev),
      dir(Gdims({n,2})){
      CPUCODE(arr=new TYPE[std::max(memsize,1)]);
      GPUCODE(CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(TYPE))));
    }

    /*
    array_pool(const TensorView<TYPE>& M):
      dir(Gdims({M.dim(0),2})),
      memsize(M.asize()),
      tail(M.asize()),
      dev(M.dev){
      CNINE_ASSRT(M.ndims()==2);
      CNINE_ASSRT(M.is_regular());
      int n0=M.dim(0);
      int n1=M.dim(1);
      for(int i=0; i<n0; i++){
	dir.set(i,0,i*n1);
	dir.set(i,1,n1);
      }
      if(dev==0){
	arr=new TYPE[std::max(memsize,1)]; 
	std::copy(M.mem(),M.mem()+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arrg,M.mem(),memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
    }

    array_pool(const Tensor<TYPE>& M):
      dir(Gdims({M.dim(0),2})),
      memsize(M.asize()),
      tail(M.asize()),
      dev(M.dev){
      CNINE_ASSRT(M.ndims()==2);
      CNINE_ASSRT(M.is_regular());
      int n0=M.dim(0);
      int n1=M.dim(1);
      for(int i=0; i<n0; i++){
	dir.set(i,0,i*n1);
	dir.set(i,1,n1);
      }
      if(dev==0){
	arr=new TYPE[std::max(memsize,1)]; 
	std::copy(M.mem(),M.mem()+memsize,arr);
      }
      if(dev==1){
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arrg,M.mem(),memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
    }
    */

  public: // ---- Static constructors ------------------------------------------------------------------------


    static array_pool<TYPE> cat(const vector<reference_wrapper<array_pool<TYPE> > >& list){
      int _dev=0; 
      if(list.size()>0) _dev=list[0].get().dev;
 
      int n=0; for(auto& p:list) n+=p.get().size();
      array_pool<TYPE> R((n));
      R.dev=_dev;
      int s=0; for(auto& p:list) s+=p.get().tail;
      R.reserve(s);

      int a=0;
      for(auto& _p:list){
	array_pool<TYPE>& p=_p.get();
	CNINE_ASSRT(p.dev==_dev);
	for(int i=0; i<p.size(); i++){
	  R.dir.set(a+i,0,R.tail+p.dir(i,0));
	  R.dir.set(a+i,1,p.dir(i,1));
	}
	if(_dev==0){
	  std::copy(p.arr,p.arr+p.tail,R.arr+R.tail);
	}
	if(_dev==1){
	  CUDA_SAFE(cudaMemcpy(R.arrg+R.tail,p.arrg,p.tail*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
	}
	a+=p.size();
	R.tail+=p.tail;
      }
      return R;
    }


  public: // ---- Memory management --------------------------------------------------------------------------


    void reserve(const int n){
      if(n<=memsize) return;
      int newsize=n;
      if(dev==0){
	TYPE* newarr=new TYPE[std::max(newsize,1)];
	if(arr){
	  std::copy(arr,arr+memsize,newarr);
	  delete[] arr;
	}
	arr=newarr;
	memsize=newsize;
      }
      if(dev==1){
	TYPE* newarrg=nullptr;
	CUDA_SAFE(cudaMalloc((void **)&newarrg, std::max(newsize,1)*sizeof(TYPE)));
	if(arrg){
	  CUDA_SAFE(cudaMemcpy(newarrg,arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
	  CUDA_SAFE(cudaFree(arrg));
	}
	arrg=newarrg;
	memsize=newsize;
      }
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    array_pool(const array_pool& x){
      CNINE_COPY_WARNING();
      dev=x.dev;
      tail=x.tail;
      memsize=tail;
      if(dev==0){
	arr=new TYPE[std::max(memsize,1)];
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	CNINE_UNIMPL();
      }
      dir=x.dir;
    }

    array_pool(array_pool&& x){
      CNINE_MOVE_WARNING();
      dev=x.dev;
      tail=x.tail; x.tail=0;
      memsize=x.memsize; x.memsize=0; 
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      dir=std::move(x.dir);
      is_view=x.is_view;
    }

    array_pool<TYPE>& operator=(const array_pool<TYPE>& x){
      CNINE_ASSIGN_WARNING();

      if(is_view){
	arr=nullptr;
	arrg=nullptr;
      }else{
	if(arr) delete[] arr; 
	arr=nullptr;
	if(arrg){CUDA_SAFE(cudaFree(arrg));}
	arrg=nullptr;
      }

      dev=x.dev;
      tail=x.tail;
      memsize=x.tail;
      dir=x.dir;
      is_view=false;

      if(dev==0){
	arr=new TYPE[std::max(memsize,1)]; 
	std::copy(x.arr,x.arr+memsize,arr);
      }
      if(dev==1){
	//cout<<12233331122<<endl;
	CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(TYPE)));
	CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice));  
      }
      return *this;
    }


    array_pool<TYPE>& operator=(array_pool<TYPE>&& x){
      CNINE_MOVEASSIGN_WARNING();
      if(!is_view){
	delete[] arr; 
	arr=nullptr;
	if(arrg){CUDA_SAFE(cudaFree(arrg));}
	arrg=nullptr;
      }
      dev=x.dev;
      arr=x.arr; x.arr=nullptr;
      arrg=x.arrg; x.arrg=nullptr;
      tail=x.tail;
      memsize=x.memsize;
      dir=std::move(x.dir);
      is_view=x.is_view;
      return *this;
    }


  public: // ---- Views --------------------------------------------------------------------------------------


    array_pool<TYPE> view(){
      array_pool<TYPE> R;
      R.dev=dev;
      R.tail=tail;
      R.memsize=memsize;
      R.arr=arr;
      R.arrg=arrg;
      //R.lookup=lookup;
      R.dir=dir;
      R.is_view=true;
      return R;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------

    /*
    template<typename U=TYPE, typename = typename std::enable_if<std::is_same<int, U>::value, U>::type>
    int_pool to_int_pool(const int _dev=0){
      CNINE_ASSRT(_dev==0);
      int N=size();
      int_pool R(N,tail);
      for(int i=0; i<N; i++)
	R.arr[i+1]=N+2+offset(i);
      R.arr[N+1]=tail;
      std::copy(arr,arr+tail,R.arr+N+2);
      if(_dev>0) R.move_to_device(_dev);
      return R;
    }
    */

    template<typename U=TYPE, typename = typename std::enable_if<std::is_same<int, U>::value, U>::type>
    Ltensor<int> to_tensor(const int _dev=0){
      CNINE_ASSRT(dev==0);
      int N=size();
      Ltensor<int> R({N+tail+2});
      R.set(0,N);
      for(int i=0; i<N; i++)
	R.set(i+1,N+2+offset(i));
      R.set(N+1,tail+N+2);
      std::copy(arr,arr+tail,R.get_arr()+N+2);
      if(_dev>0) R.move_to_device(_dev);
      return R;
    }

    
  public: // ---- Transport ----------------------------------------------------------------------------------


  array_pool(const array_pool<TYPE>& x, const int _dev): 
    dir(x.dir){
    dev=_dev;
    tail=x.tail;
    memsize=x.tail;
    if(dev==0){
      arr=new TYPE[std::max(memsize,1)];
      if(x.dev==0) std::copy(x.arr,x.arr+tail,arr);
      if(x.dev==1) CUDA_SAFE(cudaMemcpy(arr,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToHost));  
    }
    if(dev==1){
      CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(TYPE)));
      if(x.dev==0) CUDA_SAFE(cudaMemcpy(arrg,x.arr,memsize*sizeof(TYPE),cudaMemcpyHostToDevice)); 
      if(x.dev==1) CUDA_SAFE(cudaMemcpy(arrg,x.arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToDevice)); 
    }
  }


  array_pool<TYPE>& to_device(const int _dev){
      if(dev==_dev) return *this;

      if(_dev==0){
	if(dev==1){
	  //cout<<"Moving array_pool to host "<<tail<<endl;
	  memsize=tail;
	  delete[] arr;
	  arr=new TYPE[std::max(memsize,1)];
	  CUDA_SAFE(cudaMemcpy(arr,arrg,memsize*sizeof(TYPE),cudaMemcpyDeviceToHost));  
	  CUDA_SAFE(cudaFree(arrg));
	  arrg=nullptr;
	  dev=0;
	}
      }

      if(_dev>0){
	if(dev==0){
	  //cout<<"Moving array_pool to device "<<tail<<endl;
	  memsize=tail;
	  if(arrg) CUDA_SAFE(cudaFree(arrg));
	  //cout<<12233331122<j<endl;
	  CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(TYPE)));
	  CUDA_SAFE(cudaMemcpy(arrg,arr,memsize*sizeof(TYPE),cudaMemcpyHostToDevice));  
	  delete[] arr;
	  arr=nullptr;
	  dev=_dev;
	}
      }
      
      return *this;
    }


    pair<TYPE*,int*> gpu_arrs(const int _dev){
      CNINE_ASSRT(dev==0);
      if(!gpu_clone){
	gpu_clone=new array_pool<TYPE>(*this,_dev);
	gpu_clone->dir.move_to_device(_dev);
      }
      return make_pair(gpu_clone->arrg,gpu_clone->dir.arrg);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_dev() const{
      return dev;
    }

    int get_device() const{
      return dev;
    }

    int size() const{
      return dir.dim(0);
    }

    int total() const{ // this might not be the sum of the sizes if there are gaps
      return tail;
    }

    int get_tail() const{
      return tail;
    }

    // deprecated 
    TYPE* get_arr() const{
      return arr;
    }

    TYPE* get_arrg() const{
      return arrg;
    }

    int get_memsize() const{
      return memsize;
    }

    int offset(const int i) const{
      CNINE_ASSRT(i<size());
      return dir(i,0);
    }

    int size_of(const int i) const{
      CNINE_ASSRT(i<size());
      return dir(i,1);
    }

    int max_size_of(const int i) const{
      CNINE_ASSRT(i<size());
      if(i<size()-1) return dir(i+1,0)-dir(i,0);
      else return tail-dir(i,0);
    }

    TYPE operator()(const int i, const int j) const{
      CNINE_ASSRT(i<size());
      CNINE_ASSRT(j<dir(i,1));
      return arr[dir(i,0)+j];
    }

    void set(const int i, const int j, const TYPE v){
      CNINE_ASSRT(i<size());
      CNINE_ASSRT(j<dir(i,1));
      arr[dir(i,0)+j]=v;
    }

    vector<TYPE> operator()(const int i) const{
      CNINE_ASSRT(i<size());
      int addr=dir(i,0);
      int len=dir(i,1);
      vector<TYPE> R(len);
      for(int i=0; i<len; i++)
	R[i]=arr[addr+i];
      return R;
    }

    auto view_of(const int i) const -> decltype(view_of_part(*this,i) ){
      return view_of_part(*this,i);
    }
    
    void push_back(const int len){
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      dir.push_back(tail,len);
      tail+=len;
    }

    void push_back(const vector<TYPE>& v){
      int len=v.size();
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      for(int i=0; i<len; i++)
	arr[tail+i]=v[i];
      dir.push_back(tail,len);
      tail+=len;
    }
    
    /*
      void push_back(const TYPE x, const vector<TYPE>& v){
      int len=v.size()+1;
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      arr[tail]=x;
      for(int i=0; i<len-1; i++)
	arr[tail+i+1]=v[i];
      dir.push_back(tail,len);
      tail+=len;
    }
    */

    void push_back(const std::set<TYPE>& v){
      int len=v.size();
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      int i=0; 
      for(TYPE p:v){
	arr[tail+i]=p;
	i++;
      }
      dir.push_back(tail,len);
      tail+=len;
    }

    void push_back(const initializer_list<TYPE>& v){
      push_back(vector<TYPE>(v));
    }

    void push_back(const int i, const TYPE v){
      CNINE_ASSRT(i<size());
      CNINE_ASSRT(size_of(i)<max_size_of(i));
      arr[dir(i,0)+dir(i,1)]=v;
      dir.set(i,1,dir(i,1)+1);
    }

    void forall(const std::function<void(const vector<TYPE>&)>& lambda) const{
      int n=size();
      for(int i=0; i<n; i++)
	lambda((*this)(i));
    }

    void for_each(const std::function<void(const vector<TYPE>&)>& lambda) const{
      int n=size();
      for(int i=0; i<n; i++)
	lambda((*this)(i));
    }

    void for_each(const std::function<void(const int, const vector<TYPE>&)>& lambda) const{
      int n=size();
      for(int i=0; i<n; i++)
	lambda(i,(*this)(i));
    }

    void for_each(const std::function<void(const int, const TYPE&)>& lambda) const{
      int n=size();
      for(int i=0; i<n; i++){
	int m=size_of(i);
	for(int j=0; j<m; j++)
	  lambda(i,(*this)(i,j));
      }
    }

    void for_each_of(const int i, const std::function<void(const TYPE)>& lambda) const{
      CNINE_ASSRT(i<size());
      int offs=offset(i);
      int n=size_of(i);
      for(int j=0; j<n; j++)
	lambda(arr[offs+j]);
    }

    void for_each_of(const int i, std::function<void(const TYPE&)>& lambda) const{
      CNINE_ASSRT(i<size());
      int offs=offset(i);
      int n=size_of(i);
      for(int j=0; j<n; j++)
	lambda(arr[offs+j]);
    }

    vector<vector<TYPE> > as_vecs() const{
      vector<vector<TYPE> > R;
      forall([&](const vector<TYPE>& x){R.push_back(x);});
      return R;
    }

    bool operator==(const array_pool<TYPE>& y) const{
      if(size()!=y.size()) return false;
      for(int i=0; i<size(); i++){
	int n=dir(i,1);
	int offs=dir(i,0);
	int offsy=y.dir(i,0);
	if(n!=y.dir(i,1)) return false;
	for(int j=0; j<n; j++)
	  if(arr[offs+j]!=y.arr[offsy+j]) return false;
      }
      return true;
    }

    bool operator!=(const array_pool<TYPE>& y) const{
      return !((*this)==y);
    }


  public: // ---- Specialized --------------------------------------------------------------------------------

    /*
    vector<TYPE> subarray_of(const int i, const int beg) const{
      assert(i<size());
      auto& p=lookup[i];
      int addr=p.first+beg;
      int len=p.second-beg;
      assert(len>=0);
      vector<TYPE> R(len);
      for(int i=0; i<len; i++)
	R[i]=arr[addr+i];
      return R;
    }


    void push_back_cat(TYPE first, const vector<TYPE>& v){
      int len=v.size()+1;
      if(tail+len>memsize)
	reserve(std::max(2*memsize,tail+len));
      arr[tail]=first;
      for(int i=0; i<len-1; i++)
	arr[tail+1+i]=v[i];
      lookup.push_back(pair<int,int>(tail,len));
      tail+=len;
    }
    */

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "array_pool";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	auto v=(*this)(i);
	int k=v.size(); // why is this needed?
	oss<<base_indent<<"(";
	for(int j=0; j<k-1; j++){
	  oss<<v[j]<<",";
	}
	if(v.size()>0) oss<<v.back();
	oss<<")"<<endl;
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const array_pool& v){
      stream<<v.str(); return stream;}

  };


  inline Itensor1_view view_of_part(const array_pool<int>& x, const int i){
      CNINE_ASSRT(i<x.size());
      return Itensor1_view(x.arr+x.dir(i,0),x.dir(i,1),1,x.dev);
  }

  inline Rtensor1_view view_of_part(const array_pool<float>& x, const int i){
      CNINE_ASSRT(i<x.size());
      return Rtensor1_view(x.arr+x.dir(i,0),x.dir(i,1),1,x.dev);
  }


}

#endif
    //static array_pool<TYPE> cat(const initializer_list<reference_wrapper<array_pool<TYPE> > >& list){
    //return cat(vector<reference_wrapper<array_pool<TYPE> > >(list));
    //}
