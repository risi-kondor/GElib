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

#ifndef _BlockDiagonalize
#define _BlockDiagonalize

#include "Tensor.hpp"
#include "SingularValueDecomposition.hpp"


namespace cnine{


  class range_set: public set<int>{
  public:
    range_set(const int n){
      for(int i=0; i<n; i++) insert(i);
    }
  };


  template<typename TYPE>
  class ConnectedComponents: public  vector<vector<TYPE> >{
  public:

    using vector<vector<TYPE> >::back;
    using vector<vector<TYPE> >::emplace_back;

    set<TYPE> nodes;
    std::function<bool(const TYPE x, const TYPE y)> condition;


    ConnectedComponents(set<TYPE> _nodes, const std::function<bool(const TYPE x, const TYPE y)>& _condition):
      nodes(_nodes), condition(_condition){

      while(nodes.size()>0){
	TYPE x=*nodes.begin();
	nodes.erase(x);
	emplace_back();
	back().push_back(x);
	grow_component(back(),x);
      }
    }

    ConnectedComponents(const int n, const std::function<bool(const TYPE x, const TYPE y)>& _condition):
      ConnectedComponents(range_set(n),_condition){}


  private:

    void grow_component(vector<TYPE>& comp, const TYPE x){

      bool success=true;
      while(success){

	success=false;
	for(auto y:nodes)
	  if(condition(x,y)){
	    comp.push_back(y);
	    nodes.erase(y);
	    grow_component(comp,y);
	    success=true;
	    break;
	  }
      }
    }

  };


  template<typename TYPE>
  class BlockDiagonalize{
  public:

    Tensor<TYPE> U;
    Tensor<TYPE> V;
    vector<int> sizes;

    BlockDiagonalize(const TensorView<TYPE>& A, const TYPE precision=10e-5){
      CNINE_ASSRT(A.ndims()==2);
      int n=A.dims[0];
      int m=A.dims[1];
      int p=min(n,m);

      U=Tensor<TYPE>(dims(n,p));
      V=Tensor<TYPE>(dims(m,p));

      auto svd=SingularValueDecomposition(A);
      auto Um=svd.U();
      auto Vm=svd.V();
      auto Sm=svd.S();

      //cout<<Um<<endl;
      //cout<<Vm<<endl;
      //cout<<Sm<<endl;
      //print(Um*diag(Sm)*transp(Vm));

      ConnectedComponents<int> components(p,[&](const int i, const int j){
	  //cout<<inp(Um.row(i),Vm.row(j))<<endl;
	  return (abs(inp(Um.col(i),Vm.col(j)))>precision);});

      //cout<<components.size()<<" components"<<endl;
      int i=0;
      for(auto& p:components){
	sizes.push_back(p.size());
	for(auto q:p){
	  U.col(i)=Um.col(q);
	  V.col(i)=Vm.col(q);
	  i++;
	}
      }

    }

    
  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<"Blocks: ";
      for(auto& p: sizes)
	oss<<p<<" ";
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const BlockDiagonalize<TYPE>& x){
      stream<<x.str(); return stream;
    }


  };


}

#endif 

/*
      find_connected_components(singular_vectors,

      consume(singular_vectors,[&](int i){return true},
	[&](int i){
	  vector<int> clique;
	  clique.push_back(i);
	  auto u=Um.row(i);

	  consume(singular_vectors,
	    [&](int j){return inp(u,Vm.row(j))>precision;},
	    [&](int j){
	      i=j;
	      clique.push_back(i);
	      auto u=Um.row(i);
	    }

      int ndone=0;
      vector<bool> used(p);
      while(true){
	int i=0; 
	while(i<p && used[i]!=0) i++;
	if(i==p) break;

	vector<int> clique;
	clique.push_back(i);
	_U.row(ndone)=Um.row(i);
	_V.row(ndone)=Vm.row(i);
	TensorView<TYPE> u=Um.row(i);
	used[i]=true;
	ndone++;

	bool found=true;
	while(found){
	  found=false;

	  for(int j=0; j<p; j++){
	    if(!used[j] && inp(u,Vm.row(j))>precision){
	      clique.push_back(j);
	      _U.row(ndone)=Um.row(j);
	      _V.row(ndone)=Vm.row(j);
	      u=Um.row(j);
	      used[j]=true;
	      ndone++;
	      found=true;
	      break;
	    }
	  }
	}
	_sizes.push_back(clique.size());

      }

*/    //vector<vector<TYPE> > components;
      //range_set singular_vectors(p);
  /*
  template<typename TYPE>
  inline void consume(set<TYPE>& stuff, 
    std::function<bool(const TYPE& item)>& cond, 
    std::function<void(const TYPE& item)>& lambda){

    bool success=true;
    while(success){

      success=false;
      for(auto& p: stuff)
	if(cond(p)){
	  stuff.erase(p);
	  lambda(p);
	  success=true;
	  break;
	}
    }
  }
  */
