/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _CnineSparseGraph
#define _CnineSparseGraph

#include "Cnine_base.hpp"
#include <unordered_map>
#include "map_of_maps.hpp"
#include "labeled_tree.hpp"
#include "int_tree.hpp"
#include "int_pool.hpp"


namespace cnine{

  // This class represents a weighted, undirected graph

  template<typename KEY, typename TYPE, typename LABEL=int>
  class sparse_graph: public map_of_maps<KEY,KEY,TYPE>{
  public:

    typedef map_of_maps<KEY,KEY,TYPE> BASE;

    using BASE::data;

    int n=0;

    TensorView<LABEL> labels;
    bool labeled=false;

    TensorView<int> degrees;
    bool degreesp=false;


  public: // ---- Constructors -------------------------------------------------------------------------------


    sparse_graph(){};

    sparse_graph(const int _n):
      n(_n){}

    sparse_graph(const initializer_list<pair<KEY,KEY> >& list):
      sparse_graph([](const initializer_list<pair<KEY,KEY> >& list){
	  TYPE t=0; for(auto& p: list) t=std::max(std::max(p.first,p.second),t);
	  return t+1;}(list), list){}

    sparse_graph(const int _n, const initializer_list<pair<KEY,KEY> >& list): 
      sparse_graph(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    sparse_graph(const vector<pair<KEY,KEY> >& list):
      sparse_graph([](const vector<pair<KEY,KEY> >& list){
	  KEY t=0; for(auto& p: list) t=std::max(std::max(p.first,p.second),t);
	  return t+1;}(list), list){}

    sparse_graph(const int _n, const vector<pair<KEY,KEY> >& list): 
      sparse_graph(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
    }

    sparse_graph(const int _n, const vector<pair<KEY,KEY> >& list, const TensorView<LABEL>& L): 
      sparse_graph(_n){
      for(auto p:list){
	set(p.first,p.second,1.0);
	set(p.second,p.first,1.0);
      }
      labels=L;
      labeled=true;
    }


    sparse_graph(const int n, const TensorView<int>& _edges):
      sparse_graph(n){
      CNINE_ASSRT(_edges.ndims()==2);
      CNINE_ASSRT(_edges.get_dim(0)==2);
      CNINE_ASSRT(_edges.max()<n);
      int nedges=_edges.get_dim(1);
      for(int i=0; i<nedges; i++)
	set(_edges(0,i),_edges(1,i),1.0);
    }


  public: // ---- Named Constructors -------------------------------------------------------------------------


    static sparse_graph trivial(){
      return sparse_graph(1,{});}

    static sparse_graph edge(){
      return sparse_graph(2,{{0,1}});}

    static sparse_graph triangle(){
      return sparse_graph(3,{{0,1},{1,2},{2,0}});}

    static sparse_graph square(){
      return sparse_graph(4,{{0,1},{1,2},{2,3},{3,0}});}

    static sparse_graph complete(const int n){
      vector<pair<int,int> > v;
      for(int i=0; i<n; i++)
	for(int j=0; j<i; j++)
	  v.push_back(pair<int,int>(i,j));
      return sparse_graph(n,v);
    }

    static sparse_graph cycle(const int n){
      vector<pair<int,int> > v;
      for(int i=0; i<n-1; i++)
	v.push_back(pair<int,int>(i,i+1));
      v.push_back(pair<int,int>(n-1,0));
      return sparse_graph(n,v);
    }

    static sparse_graph star(const int m){
      vector<pair<int,int> > v(m);
      for(int i=0; i<m; i++)
	v[i]=pair<int,int>(0,i+1);
      return sparse_graph(m+1,v);
    }

    static sparse_graph random(const int _n, const float p=0.5){
      sparse_graph G(_n);
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<_n; i++) 
	for(int j=0; j<i; j++)
	  if(distr(rndGen)<p){
	    G.set(i,j,1.0);
	    G.set(j,i,1.0);
	  }
      return G;
    }


  public: // ---- Conversions ------------------------------------------------------------------------------


    sparse_graph(const TensorView<TYPE>& x){
      CNINE_ASSRT(x.ndims()==2);
      CNINE_ASSRT(x.dim(0)==x.dim(1));
      n=x.dim(0);
      for(int i=0; i<n; i++)
	for(int j=0; j<=i; j++)
	  if(x(i,j)) set(i,j,x(i,j));
    }

    TensorView<TYPE> dense() const{
      auto R=TensorView<TYPE>::zero({n,n});
      BASE::for_each([&](const KEY& i, const KEY& j, const TYPE& v){
	  R.set(i,j,v);});
      return R;
    }


    int_pool as_int_pool() const{
      int_pool R(n,2*nedges());
      int* arr=R.arr;

      int tail=n+2;
      arr[1]=tail;

      for(int i=0; i<n; i++){
	auto it=data.find(i);
	if(it!=data.end())
	  for(auto p:it->second)
	    arr[tail++]=p.first;
	arr[i+2]=tail;
      }

      return R;
    }

    sparse_graph(const int_pool& x){
      n=x.getn();
      for(int i=0; i<n; i++){
	int m=x.size_of(i);
	for(int j=0; j<m; j++)
	  set(i,x(i,j),1.0);
      }
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getn() const{
      return n;
    }

    int nedges() const{
      return BASE::size()/2;
    }

    bool is_labeled() const{
      return labeled;
    }

    bool with_degrees() const{
      return degreesp;
    }

    int nneighbors(const int i) const{
      return data[i].size();
    }

    bool is_neighbor(const KEY& i, const KEY& j) const{
      return is_filled(i,j);
    }
    
    void set(const KEY& i, const KEY& j, const TYPE& v){
      BASE::set(i,j,v);
      BASE::set(j,i,v);
    }

    void set_labels(const TensorView<int>& _labels){
      CNINE_ASSRT(_labels.ndims()==1);
      CNINE_ASSRT(_labels.dim(0)==n);
      labels.reset(_labels);
      labeled=true;
    }

    void set_degrees(const TensorView<int>& _degrees){
      CNINE_ASSRT(_degrees.ndims()==1);
      CNINE_ASSRT(_degrees.dim(0)==n);
      degrees.reset(_degrees);
      degreesp=true;
    }

    vector<int> neighbors(const int i) const{
      vector<int> r;
      const auto _r=data[i];
      for(auto& p: _r)
	r.push_back(p.first);
      return r;
    }

    template<typename TYPE2>
    void insert(const sparse_graph<KEY,TYPE2>& H, const vector<int>& v){
      for(auto p:v)
	CNINE_ASSRT(p<n);
      H.for_each_edge([&](const int i, const int j, const TYPE2& val){
	  set(v[i],v[j],val);});
    }

    bool operator==(const sparse_graph& x) const{
      if(n!=x.n) return false;
      if(labeled!=x.labeled) return false; 
      if(labeled && (labels!=x.labels)) return false;
      if(degreesp!=x.degreesp) return false; 
      if(degreesp && (degrees!=x.degrees)) return false;
      return BASE::operator==(x);
    }

    

  public: // ---- Lambdas ----------------------------------------------------------------------------------


    void for_each_edge(std::function<void(const int, const int)> lambda, const bool self=0) const{
      BASE::for_each([&](const int i, const int j, const TYPE& v){if(i<=j) lambda(i,j);});
    }

    void for_each_edge(std::function<void(const int, const int, const TYPE&)> lambda, const bool self=0) const{
      BASE::for_each([&](const int i, const int j, const TYPE& v){if(i<=j) lambda(i,j,v);});
    }


  public: // ---- Subgraphs ----------------------------------------------------------------------------------


    labeled_tree<KEY> greedy_spanning_tree(const KEY root=0) const{
      //CNINE_ASSRT(root<n);
      labeled_tree<KEY> r(root);
      vector<bool> matched(n,false);
      matched[root]=true;
      for(auto& p: BASE::data[root]){
	if(!p.second) continue;
	if(matched[p.first]) continue;
	matched[p.first]=true;
	r.children.push_back(greedy_spanning_tree(p.first,matched));
      }
      return r;
    }

    labeled_tree<KEY>* greedy_spanning_tree(const int v, vector<bool>& matched) const{
      labeled_tree<KEY>* r=new labeled_tree<KEY>(v);
      for(auto& p: BASE::data[v]){
	if(!p.second) continue;
	if(matched[p.first]) continue;
	matched[p.first]=true;
	r->children.push_back(greedy_spanning_tree(p.first,matched));
      }
      return r;
    }


    /*
    int_tree spanning_tree_as_int_tree(int root=0) const{
      int_tree r;
      vector<bool> matched(n,false);

      matched[root]=true;
      int m=0; 
      for(auto& p: BASE::data[root]){
	m++;}
      auto root_node=r.add_root(root,m);
 
      int i=0;
      for(auto& p: BASE::data[root])
	spanning_tree_as_int_tree(root_node,i++,p.first,matched);
      return r;
    }
    
    void spanning_tree_as_int_tree(int_tree::node& parent, int i, int v, vector<bool>& matched) const{
      matched[v]=true;
      int m=0; 
      for(auto& p: BASE::data[v])
	if(!matched[p.first]) m++;
      auto node=parent.add_child(i,v,m);

      int j=0;
      for(auto& p: BASE::data[v])
	if(!matched[p.first])
	  spanning_tree_as_int_tree(node,j++,p.first,matched);
    }
    */


  private:


  public: // ---- I/O -----------------------------------------------------------------------------------------


    string classname() const{
      return "sparse_graph";
    }

    string str(const string indent="") const{
      ostringstream oss;
      //oss<<indent<<"Graph with "<<n<<" vertices:"<<endl;
      oss<<dense().str(indent); //+"  ");
      //if(is_labeled()) oss<<indent<<"L:"<<labels.str()<<endl;
      //if(with_degrees()) oss<<indent<<"D:"<<degrees.str()<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const sparse_graph& x){
      stream<<x.str(); return stream;}


  };

}


namespace std{

  template<typename KEY, typename TYPE, typename LABEL>
  struct hash<cnine::sparse_graph<KEY,TYPE,LABEL> >{
  public:
    size_t operator()(const cnine::sparse_graph<KEY,TYPE,LABEL>& x) const{
      return hash<cnine::map_of_maps<KEY,KEY,TYPE> >()(x);
      //      if(x.is_labeled()) return (hash<cnine::SparseRmatrix>()(x)<<1)^hash<cnine::RtensorA>()(x.labels);
      //return hash<cnine::SparseRmatrix>()(x);
    }
  };
}




#endif 
    /*
      int_pool pool=as_int_pool();

      vector<int> addr(n,-1);
      vector<int> kids_visited(n,0);
      vector<int> parent(n);

      compact_int_tree r;
      //auto_vector<int> kids;
      int tail=0;
      int loc=0;
      int nvisited=0;

      while(nvisited<n){

	if(addr[node]==-1){ //first time seeing this node
	  assert(loc==tail);
	  r.arr[loc]=node;
	  addr[node]=loc;
	  nvisited++;
	  tail+=2;

	  for(int i=0; i<pool.size_of(node); i++){
	    int candidate=pool(node,i);
	    if(addr[candidate]<0){
	      r.arr[tail++]=candidate; // temporary
	      addr[candidate]=0; //temporary
	      parent[candidate]=node;
	    }
	  }
	  r.arr[loc+1]=tail-loc-2; // number of kids
	}

	if(kids_visited[node]<R.arr[]){
	  
	}
    */
	/*
	int total_children=pool.size_of(node);
	for(;nchecked[node]<total_children; nchecked[node]++){ // find next child that is new
	  if(addr[pool(node,nchecked[node])]==-1) break;
	}

	if(nchecked[node]<total_children){ // add child as new node
	  r[loc]+nchildren[node]=tail;
	  node=pool(node,nchecked[node]);
	  nchildren[node]++;
	  loc=tail;
	}else{
	}
	*/
    /* unused 
    array_pool<int> as_array_pool() const{
      array_pool<int> R(n,2*nedges());
      int* arr=R.arr;

      int tail=n+2;
      arr[1]=tail;

      for(int i=0; i<n; i++){
	auto it=data.find(i);
	if(it!=data.end())
	  for(auto p:it->second)
	    arr[tail++]=p.first;
	arr[i+2]=tail;
      }

      return R;
    }
    */ 

