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

#ifndef _GatherMapB
#define _GatherMapB

#include "Cnine_base.hpp"
#include "hlists.hpp"
#include "FixedkGatherMap.hpp"
#include "map_of_lists.hpp"
#include "fnlog.hpp"
#include "RemoteCopy.hpp"

namespace cnine{

  extern CnineLog cnine_log;


  class GatherMapB{
  private:
  public:

    typedef cnine::TensorView<int> ITENSOR;

    hlists<int> arr;
    shared_ptr<GatherMapB> _inv;
    mutable bool sorted=false;

    int n_out=0;
    int n_in=0;
    //int* arrg=nullptr; // unsafe!!

    //cnine::monitored<cnine::Ltensor<int> > gpu_format=
    //cnine::monitored<cnine::Ltensor<int> >([this](){
    //  return to_share(new cnine::Ltensor<int>(arr.to_tensor(1)));});

    RemoteCopy<int,ITENSOR> on_device=cnine::RemoteCopy<int,ITENSOR>([this](const int& _dev){
	//cout<<arr.to_tensor(0)<<endl;
	return to_share(new ITENSOR(arr.to_tensor(_dev)));});


  public:

    vector<shared_ptr<FixedkGatherMap> > fixedk_maps;

    int in_columns=1;
    int out_columns=1;
    int in_columns_n=1;
    int out_columns_n=1;


  public:

    virtual ~GatherMapB(){
      //if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    GatherMapB(){}

    GatherMapB(const int _n_out, const int _n_in): 
      n_out(_n_out), n_in(_n_in){}

    GatherMapB(const vector<int>& sources, const vector<int>& targets){
      cnine::fnlog timer("GatherMapB::GatherMapB(const vector<int>& sources, const vector<int>& targets)");
      CNINE_ASSRT(sources.size()==targets.size());

      int N=sources.size();
      unordered_map<int,int> sizes;
      for(int i=0; i<N; i++){
	sizes[targets[i]]++;
	n_out=std::max(n_out,targets[i]+1);
	n_in=std::max(n_in,sources[i]+1);
      }

      int n=sizes.size();
      vector<int> heads(n);
      vector<int> lengths(n);
      unordered_map<int,int> mapping;
      int i=0;
      for(auto p:sizes){
	heads[i]=p.first;
	lengths[i]=p.second;
	mapping[p.first]=i;
	i++;
      }

      arr=hlists<int>(heads,lengths);
      for(int i=0; i<N; i++){
	arr.push_back(mapping[targets[i]],sources[i]);
      }
    }
    

    explicit GatherMapB(const TensorView<int>& M, int n_in=-1, int n_out=-1){

      CNINE_ASSRT(M.ndims()==2);
      CNINE_ASSRT(M.dims(1)==2);
      int N=M.dim(0);
      if(n_in==-1)
	for(int i=0; i<N; i++)
	  n_in=std::max(n_in,M(i,0)+1);
      if(n_out==-1)
	for(int i=0; i<N; i++)
	  n_out=std::max(n_out,M(i,1)+1);

      unordered_map<int,int> sizes;
      for(int i=0; i<N; i++){
	CNINE_ASSRT(M(i,0)<n_in);
	CNINE_ASSRT(M(i,1)<n_out);
	sizes[M(i,1)]++;
      }

      int n=sizes.size();
      vector<int> heads(n);
      vector<int> lengths(n);
      unordered_map<int,int> mapping;
      int i=0;
      for(auto p:sizes){
	heads[i]=p.first;
	lengths[i]=p.second;
	mapping[p.first]=i;
	i++;
      }

      arr=hlists<int>(heads,lengths,fill_noalloc());
      for(int i=0; i<N; i++){
	arr.push_back(mapping[M(i,1)],M(i,0));
      }
    }
    

    GatherMapB(const map_of_lists<int,int>& x, const int _out_columns=1, const int _in_columns=1,
      const int _out_columns_n=1, const int _in_columns_n=1):
      in_columns(_in_columns),
      out_columns(_out_columns),
      in_columns_n(_in_columns_n),
      out_columns_n(_out_columns_n){
      cnine::fnlog timer("GatherMapB::GatherMapB(const map_of_lists<int,int>& map)");
      //cout<<"make GatherMapB"<<endl;

      int total=0;
      for(auto& p:x){
	total+=p.second.size();
	bump(n_out,p.first+1);
	for(auto& q:p.second)
	  bump(n_in,q+1);
      }

      arr.reserve(x.size()+total);
      for(auto& p:x)
	arr.push_back(p.first,p.second);
    }


    // this is specifically for BlockCsparseMatrix, maybe move it there?
    GatherMapB(const map<int,std::map<int,int> >& x, const int _out_columns=1, const int _in_columns=1,
      const int _out_columns_n=1, const int _in_columns_n=1):
      in_columns(_in_columns),
      out_columns(_out_columns),
      in_columns_n(_in_columns_n),
      out_columns_n(_out_columns_n){

      int total=0;
      for(auto& p:x){
	total+=p.second.size();
	bump(n_out,p.first+1);
	for(auto& q:p.second)
	  bump(n_in,q.first+1);
      }

      arr.reserve(x.size()+total);
      for(auto& p:x){
	vector<int> v(p.second.size());
	int i=0;
	for(auto& q:p.second)
	  v[i++]=q.first;
	arr.push_back(p.first,v);
      }
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    //GatherMapB(const int _n, const array_pool<int>& _arr):
    //arr(_arr), n(_n){
    //}

    //GatherMapB(const int _n, array_pool<int>&& _arr):
    //arr(std::move(_arr)),  n(_n){
    //}


  public: // ---- Named constructors -------------------------------------------------------------------------


    static GatherMapB random(const int n, const int m, const float p=0.5){
      GatherMapB r(n,m);
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<n; i++){
	vector<int> v;
	for(int j=0; j<m; j++)
	  if(distr(rndGen)<p)
	    v.push_back(j);
	r.arr.push_back(i,v);
      }
      return r;
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    GatherMapB(const GatherMapB& x, const int _dev):
      arr(x.arr,_dev), n_out(x.n_out), n_in(x.n_in){
    }

    GatherMapB& move_to_device(const int _dev){
      arr.to_device(_dev);
      return *this;
    }

    //[[deprecated]]
    //int* get_arrg(const int _dev=1) const{
    //if(!arrg) make_arrg();
    //return arrg;
    //}

    //[[deprecated]]
    /*
    void make_arrg() const{
      cnine::fnlog timer("GatherMapB::make_arrg()");
      //cout<<arr.dir.memsize<<"...."<<arr.get_memsize()<<endl;
      //int memsize=arr.get_memsize()+arr.dir.memsize;
      int memsize=arr.get_tail()+arr.size()*2;
      //cout<<"GatherMapB::make_arrg()"<<endl;
      CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
      CUDA_SAFE(cudaMemcpy(arrg, arr.dir.arr, 2*arr.size()*sizeof(int),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(arrg+2*arr.size(), arr.arr, arr.get_tail()*sizeof(int),cudaMemcpyHostToDevice));  
    }
    */


  public: // ---- Getters ------------------------------------------------------------------------------------


    int get_dev() const{
      return arr.get_dev();
    }

    int get_nout() const{
      return n_out;
    }

    int get_nin() const{
      return n_in;
    }

    int size() const{
      return arr.size();
    }

    // need at least one virtual fn for class 
    // to be polymorphic 
    virtual int n_ops() const{
      return arr.get_tail()-arr.size();
    }

    int offset(const int i) const{
      return arr.offset(i);
    }

    int size_of(const int i) const{
      return arr.size_of(i);
    }

    int target(const int i) const{
      return arr.head(i);
    }

    void set_target(const int i, const int x){
      arr.set_head(i,x);
    }

    int operator()(const int i, const int j) const{
      return arr(i,j);
    }

    shared_ptr<GatherMapB> inv_ptr() const{
      if(!_inv.get()) make_inv();
      return _inv;
    }

    const GatherMapB& inv() const{
      if(!_inv.get()) make_inv();
      return *_inv;
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each(std::function<void(const int i, const int j)> lambda) const{
      //arr.for_each(lambda); // just use this?
      int N=size();
      for(int i=0; i<N; i++){
	int M=size_of(i);
	int targt=target(i);
	for(int j=0; j<M; j++)
	  lambda(targt,(*this)(i,j));
      }
    }


  public: // ---- Setters ------------------------------------------------------------------------------------

    
    void set(const int i, const int j, const int x){
      arr.set(i,j,x);
    }

    int push_back(const int len){
      sorted=false;
      arr.push_back(len);
      return size()-1;
    }

    void push_back(const int t, const std::set<int>& v){
      sorted=false;
      arr.push_back(t,v);
    }

    void push_back(const int t, const vector<int>& v){
      sorted=false;
      arr.push_back(t,v);
    }

    void push_back(const int t, const initializer_list<int>& v){
      sorted=false;
      arr.push_back(t,v);
    }



  public: // ---- Operations ---------------------------------------------------------------------------------


    void make_inv() const{
      cnine::fnlog timer("GatherMapB::make_inv()");
      map<int,vector<int> > inv_map;
      int total=0;
      for_each([&](const int i, const int j){
	  inv_map[j].push_back(i);
	  total++;
	});
      GatherMapB* r=new GatherMapB(n_in,n_out);
      //if(inv_map.size()==0) r=new GatherMapB(0); 
      //else r=new GatherMapB(inv_map.rbegin()->first+1);
      r->arr.reserve(inv_map.size()+total);
      for(auto& p: inv_map)
	r->arr.push_back(p.first,p.second);
      r->in_columns=out_columns;
      r->out_columns=in_columns;
      r->in_columns_n=out_columns_n;
      r->out_columns_n=in_columns_n;

      const_cast<GatherMapB&>(*this)._inv.reset(r);
    }

    
    const GatherMapB& sort() const{
      cnine::fnlog timer("GatherMapB::sort()");
      if(sorted) return *this;

      map<int,vector<int> > lengths;
      int N=size();
      for(int i=0; i<N; i++)
	lengths[-size_of(i)].push_back(i);
      GatherMapB r(n_out,n_in);
      r.arr.reserve(arr.tail);
      for(auto& p:lengths){
	int K=-p.first;
	for(auto q:p.second){
	  int i=r.push_back(K);
	  r.set_target(i,target(q));
	  for(int a=0; a<K; a++){
	    r.set(i,a,(*this)(q,a));
	  }
	}
      }
      const_cast<GatherMapB&>(*this).arr=std::move(r.arr);
      sorted=true;
      return *this;
    }


    const GatherMapB& grade(const int min_size=0) const{
      cnine::fnlog timer("GatherMapB::grade()");
      map<int,vector<int> > lengths;
      int N=size();
      for(int i=0; i<N; i++)
	lengths[size_of(i)].push_back(i);

      int rem_size=0;
      for(auto& p:lengths)
	if(p.second.size()<min_size)
	  rem_size+=(p.first+1)*p.second.size();

      GatherMapB r(n_out,n_in);
      r.arr.reserve(rem_size);

      for(auto& p:lengths){
	int K=p.first;
	if(p.second.size()>=min_size){
	  FixedkGatherMap* g=new FixedkGatherMap(p.second.size(),K);
	  auto gv=g->view2();
	  for(int j=0; j<p.second.size(); j++)
	    gv.slice0(j)=arr.array_pool<int>::view_of(p.second[j]);
	  const_cast<GatherMapB&>(*this).fixedk_maps.push_back(shared_ptr<FixedkGatherMap>(g));
	}else{
	  for(auto q:p.second){
	    int i=r.push_back(K);
	    r.set_target(i,target(q));
	    for(int a=0; a<K; a++)
	      r.set(i,a,(*this)(q,a));
	  }
	}
      }

      const_cast<GatherMapB&>(*this).arr=std::move(r.arr);
      return *this;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GatherMapB";
    }

    string repr() const{
      return "GatherMapB";
    }

    string str(const string indent="") const{
      //for(int i=0; i<arr.tail; i++) cout<<arr.arr[i]<<" "; cout<<endl;
      //cout<<arr.dir<<endl;
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<target(i)<<"<-(";
	for(int j=0; j<size_of(i); j++){
	  oss<<(*this)(i,j)<<",";
	}
	if(size_of(i)>0) oss<<"\b";
	oss<<")\n";
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const GatherMapB& v){
      stream<<v.str(); return stream;}

  };



}

#endif 
