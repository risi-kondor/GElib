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

#ifndef _WeightedGatherMapB
#define _WeightedGatherMapB

#include "Cnine_base.hpp"
#include "hlists.hpp"
#include "FixedkGatherMap.hpp"
#include "map_of_lists.hpp"
#include "fnlog.hpp"
#include "GatherMapB.hpp"

namespace cnine{

  extern CnineLog cnine_log;


  class WeightedGatherMapB: public GatherMapB{
  private:
  public:

    typedef GatherMapB BASE;

    using BASE::BASE;

    //hlists<int> arr;
    shared_ptr<WeightedGatherMapB> _inv;
    //mutable bool sorted=false;

    //int n=0;
    //int* arrg=nullptr; // unsafe!!

  public:

    vector<shared_ptr<FixedkGatherMap> > fixedk_maps;

    //int in_columns=1;
    //int out_columns=1;


  public:

    ~WeightedGatherMapB(){
      //if(arrg) {CUDA_SAFE(cudaFree(arrg));}
    }


  public: // ---- Constructors -------------------------------------------------------------------------------


    WeightedGatherMapB(){}

    WeightedGatherMapB(const int _n_out, const int _n_in): 
      BASE(_n_out,_n_in){}

    WeightedGatherMapB(const vector<int>& sources, const vector<int>& targets, const vector<float>& weights){
      cnine::fnlog timer("WeightedGatherMapB::WeightedGatherMapB(const vector<int>& sources, const vector<int>& targets)");
      CNINE_ASSRT(sources.size()==targets.size());
      // set n_out and n_in

      int N=sources.size();
      unordered_map<int,int> sizes;
      for(int i=0; i<N; i++)
	sizes[targets[i]]++;
      
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
	push_back(mapping[targets[i]],sources[i],weights[i]);
      }
    }
    

    WeightedGatherMapB(const map_of_lists<int,int>& x, const int _out_columns=1, const int _in_columns=1){
      in_columns=_in_columns;
      out_columns=_out_columns;
      cnine::fnlog timer("WeightedGatherMapB::WeightedGatherMapB(const map_of_lists<int,int>& map)");
      //cout<<"make WeightedGatherMapB"<<endl;

      int total=0;
      for(auto& p:x)
	total+=p.second.size();

      arr.reserve(x.size()+total);
      for(auto& p:x)
	arr.push_back(p.first,p.second);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    //WeightedGatherMapB(const int _n, const array_pool<int>& _arr):
    //arr(_arr), n(_n){
    //}

    //WeightedGatherMapB(const int _n, array_pool<int>&& _arr):
    //arr(std::move(_arr)),  n(_n){
    //}


  public: // ---- Named constructors -------------------------------------------------------------------------


    static WeightedGatherMapB random(const int _n_out, const int _n_in, const float p){
      WeightedGatherMapB r(_n_out,_n_in);
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<_n_out; i++){
	vector<int> v;
	for(int j=0; j<_n_in; j++)
	  if(distr(rndGen)<p)
	    v.push_back(j);
	r.arr.push_back(i,v);
      }
      return r;
    }


  public: // ---- Transport ----------------------------------------------------------------------------------


    //WeightedGatherMapB(const WeightedGatherMapB& x, const int _dev):
    //arr(x.arr,_dev), BASE(x.n){
    //}

    WeightedGatherMapB& move_to_device(const int _dev){
      arr.to_device(_dev);
      return *this;
    }

    //int* get_arrg(const int _dev=1) const{
    //if(!arrg) make_arrg();
    //return arrg;
    //}

    /*
    void make_arrg() const{
      //cout<<arr.dir.memsize<<"...."<<arr.get_memsize()<<endl;
      //int memsize=arr.get_memsize()+arr.dir.memsize;
      int memsize=arr.get_tail()+arr.size()*2;
      CUDA_SAFE(cudaMalloc((void **)&arrg, std::max(memsize,1)*sizeof(int)));
      CUDA_SAFE(cudaMemcpy(arrg, arr.dir.arr, 2*arr.size()*sizeof(int),cudaMemcpyHostToDevice));  
      CUDA_SAFE(cudaMemcpy(arrg+2*arr.size(), arr.arr, arr.get_tail()*sizeof(int),cudaMemcpyHostToDevice));  
    }
    */

  public: // ---- Access -------------------------------------------------------------------------------------


    //int get_dev() const{
    //return arr.get_dev();
    //}

    //int getn() const{
    //return n;
    //}

    int size() const{
      return arr.size();
    }

    int n_ops() const{
      return arr.get_tail()/2-arr.size();
    }

    //int offset(const int i) const{
    //return arr.offset(i);
    //}

    int size_of(const int i) const{
      return arr.size_of(i)/2;
    }

    //int target(const int i) const{
    //return arr.head(i);
    //}

    //void set_target(const int i, const int x){
    //arr.set_head(i,x);
    //}

    pair<int,float> operator()(const int i, const int j) const{
      return make_pair(arr(i,2*j),reinterpret_cast<const float&>(arr.ref(i,2*j+1)));
    }
    
    void set(const int i, const int j, const int x, const float& c){
      arr.set(i,2*j,x);
      arr.set(i,2*j+1,reinterpret_cast<const float&>(c));
    }

    int src(const int i, const int j) const{
      return arr(i,2*j);
    }

    float weight(const int i, const int j) const{
      return reinterpret_cast<const float&>(arr.ref(i,2*j+1));
    }

    int push_back(const int len){
      sorted=false;
      arr.push_back(2*len);
      return size()-1;
    }

    void push_back(const int i, const int s, const float& w){
      arr.push_back(i,s);
      arr.push_back(i,reinterpret_cast<const int&>(w));
    }

    void push_back(const int t, const vector<int>& v){
      sorted=false;
      arr.push_back(t,v);
    }

    void for_each(std::function<void(const int i, const int j, const float w)> lambda) const{
      int N=size();
      for(int i=0; i<N; i++){
	int M=size_of(i);
	int targt=target(i);
	for(int j=0; j<M; j++)
	  lambda(targt,src(i,j),weight(i,j));
      }
    }

    shared_ptr<WeightedGatherMapB> inv_ptr() const{
      if(!_inv.get()) make_inv();
      return _inv;
    }

    const WeightedGatherMapB& inv() const{
      if(!_inv.get()) make_inv();
      return *_inv;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    void make_inv() const{
      cnine::fnlog timer("WeightedGatherMapB::make_inv()");
      map<int,vector<int> > inv_map;
      int total=0;
      for_each([&](const int i, const int j, const float v){
	  //inv_map[j].push_back(make_pair(i,v));
	  inv_map[j].push_back(i);
	  inv_map[j].push_back(*reinterpret_cast<const int*>(&v));
	  total+=2;
	});
      WeightedGatherMapB* r=new WeightedGatherMapB(n_in,n_out);
      //WeightedGatherMapB* r=new WeightedGatherMapB(inv_map.rbegin()->first+1);
      r->arr.reserve(size()+total);
      for(auto& p: inv_map)
	r->arr.push_back(p.first,p.second);
      const_cast<WeightedGatherMapB&>(*this)._inv.reset(r);
    }

    
    const WeightedGatherMapB& sort() const{
      cnine::fnlog timer("WeightedGatherMapB::sort()");
      if(sorted) return *this;

      map<int,vector<int> > lengths;
      int N=size();
      for(int i=0; i<N; i++)
	lengths[-size_of(i)].push_back(i);
      WeightedGatherMapB r(n_out,n_in);
      r.arr.reserve(arr.tail);
      for(auto& p:lengths){
	int K=-p.first;
	for(auto q:p.second){
	  int i=r.push_back(K);
	  r.set_target(i,target(q));
	  for(int a=0; a<K; a++){
	      r.set(i,a,src(q,a),weight(q,a));
	  }
	}
      }
      const_cast<WeightedGatherMapB&>(*this).arr=std::move(r.arr);
      sorted=true;
      return *this;
    }

    /*
    const WeightedGatherMapB& grade(const int min_size=0) const{
      cnine::fnlog timer("WeightedGatherMapB::grade()");
      map<int,vector<int> > lengths;
      int N=size();
      for(int i=0; i<N; i++)
	lengths[size_of(i)].push_back(i);

      int rem_size=0;
      for(auto& p:lengths)
	if(p.second.size()<min_size)
	  rem_size+=(p.first+1)*p.second.size();

      WeightedGatherMapB r(n);
      r.arr.reserve(rem_size);

      for(auto& p:lengths){
	int K=p.first;
	if(p.second.size()>=min_size){
	  FixedkGatherMap* g=new FixedkGatherMap(p.second.size(),K);
	  auto gv=g->view2();
	  for(int j=0; j<p.second.size(); j++)
	    gv.slice0(j)=arr.array_pool<int>::view_of(p.second[j]);
	  const_cast<WeightedGatherMapB&>(*this).fixedk_maps.push_back(shared_ptr<FixedkGatherMap>(g));
	}else{
	  for(auto q:p.second){
	    int i=r.push_back(K);
	    r.set_target(i,target(q));
	    for(int a=0; a<K; a++)
	      r.set(i,a,(*this)(q,a));
	  }
	}
      }

      const_cast<WeightedGatherMapB&>(*this).arr=std::move(r.arr);
      return *this;
    }
    */

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "WeightedGatherMapB";
    }

    string repr() const{
      return "WeightedGatherMapB";
    }

    string str(const string indent="") const{
      //for(int i=0; i<arr.tail; i++) cout<<arr.arr[i]<<" "; cout<<endl;
      //cout<<arr.dir<<endl;
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<indent<<target(i)<<"<-(";
	for(int j=0; j<size_of(i); j++){
	  oss<<"("<<src(i,j)<<","<<weight(i,j)<<"),";
	}
	if(size_of(i)>0) oss<<"\b";
	oss<<")\n";
      }
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const WeightedGatherMapB& v){
      stream<<v.str(); return stream;}

  };



}

#endif 
