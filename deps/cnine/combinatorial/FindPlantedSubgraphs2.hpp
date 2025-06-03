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

#ifndef _CnineFindPlantedSubgraphs2
#define _CnineFindPlantedSubgraphs2

#include "Cnine_base.hpp"
#include "sparse_graph.hpp"
#include "labeled_tree.hpp"
#include "labeled_forest.hpp"
#include "TensorView.hpp"
#include "flog.hpp"


namespace cnine{

#ifdef _WITH_CUDA
  extern Tensor<int> FindPlantedSubgraphs_cu(const int_pool&, int_pool&, const cudaStream_t&);
#endif 


  template<typename LABEL=int>
  class FindPlantedSubgraphs2{
  public:

    typedef int_pool Graph;
    //typedef labeled_tree<int> labeled_tree;
    //typedef labeled_forest<int> labeled_forest;


    //const Graph& G;
    //const Graph& H;
    //const sparse_graph<int,float,LABEL>& H;
    int n;
    int nmatches=0;
    //vector<pair<int,int> > Htraversal;
    //vector<int> assignment;
    TensorView<int> matches;

    int level;
    //vector<int> matching;
    //vector<int> pseudo_iterators;


  public:


    FindPlantedSubgraphs2(const sparse_graph<int,float,LABEL>& G, const sparse_graph<int,float,LABEL>& H, const int _dev=0):
      FindPlantedSubgraphs2(G.as_int_pool(),H.as_int_pool(),_dev){}


    FindPlantedSubgraphs2(const Graph& G, Graph& H, const int _dev=0):
      //G(_G), 
      n(H.getn()),
      matches({10,H.getn()},fill_zero()){

      if(_dev>0){
	CUDA_STREAM(matches=FindPlantedSubgraphs_cu(G,H,stream));
	return;
      }

      sparse_graph<int,float,LABEL> Hsg(H);
      int_tree Htree=Hsg.greedy_spanning_tree().as_int_tree();
      //int_tree Htree=sparse_graph<int,float,LABEL>(H).spanning_tree_as_int_tree();
      //int_tree Htree=int_tree::spanning_tree(H);
      //cout<<Htree.auto_array<int>::str()<<endl;
      //cout<<Htree<<endl;
      vector<int> Htraversal=Htree.depth_first_traversal();
      //for(int i=0; i<Htraversal.size(); i++) cout<<Htraversal[i]<<" "; cout<<endl;
      vector<int> parent_of(n);
      Htree.traverse([&](const int_tree::node& x){
	  if(x.parent()>=0) 
	    parent_of[x.label()]=Htree.node_at(x.parent()).label();
	});
      //for(int i=0; i<n; i++) cout<<parent_of[i]; cout<<endl;
      vector<int> matching(n,-1);
      vector<int> pseudo_iterators(n,0);

      for(int i=0; i<G.getn(); i++){

	int level=0;
	int w=Htraversal[0];
	int v=i;

	while(level>=0){

	  int m1=H.size_of(w);
	  int m2=G.size_of(v);
	  bool success=true;

	  // check that every neighbor of w that is already part 
	  // of the matching corresponds to a neighbor of v 
	  for(int j=0; j<m1; j++){
	    int y=H(w,j);
	    if(matching[y]==-1) continue;
	    int vdash=matching[y];

	    bool found=false;
	    for(int p=0; p<m2; p++)
	      if(G(v,p)==vdash){
		found=true;
		break;
	      }
	    if(!found){
	      success=false;
	      break;
	    }
	  }

	  // check that every neighbor of v that is already part 
	  // of the matching corresponds to a neighbor of w 
	  if(success){
	    for(int j=0; j<m2; j++){
	      int vdash=G(v,j);
	      int wdash=-1;
	      for(int p=0; p<n; p++)
		if(matching[p]==vdash){
		  wdash=p;
		  break;
		}
	      if(wdash>=0){
		bool found=false;
		for(int p=0; p<m1; p++)
		  if(H(w,p)==wdash){
		    found=true;
		    break;
		  }
		if(!found){
		  success=false;
		  break;
		}
	      }
	    }
	  }

	  // if w has been successfully matched to v
	  if(success){
	    //cout<<"matched "<<w<<" to "<<v<<" at level "<<level<<endl;
	    matching[w]=v;
	    //matched[level]=v;
	    if(level==n-1){
	      add_match(matching);
	      success=false;
	    }
	  }

	  // if matched and not at the final level, try to descend
	  // even farther
	  if(success){
	    //auto hnode=Htree.node_at(Htraversal[level+1]);
	    //int neww=hnode.label();
	    //int parent_node=Htree.node_at(hnode.parent());
	    //int parentv=matching[parent_node.label()];
	    int parentv=matching[parent_of[Htraversal[level+1]]];
	    CNINE_ASSRT(parentv!=-1);
	    pseudo_iterators[level]=0;
	    int m3=G.size_of(parentv);
	    int newv=-1;
	    for(int j=0; j<m3; j++){
	      int candidate=G(parentv,j);
	      bool found=false; 
	      for(int p=0; p<n; p++)
		if(matching[p]==candidate){
		  found=true;
		  break;
		}
	      if(!found){
		newv=candidate;
		pseudo_iterators[level]=j+1;
		break;
	      }
	    }
	    if(newv>=0){
	      w=Htraversal[level+1];
	      v=newv;
	      level++;
	    }else{
	      success=false;
	    }
	  }

	  // if no match or could not descend farther forced to climb back
	  // and find alternative paths
	  if(!success){
	    matching[w]=-1;
	    level--;

	    while(level>=0){
	      int neww=Htraversal[level+1];
	      int parentv=matching[parent_of[neww]];
	      CNINE_ASSRT(parentv!=-1);
	      int m3=G.size_of(parentv);
	      int newv=-1;
	      for(int j=pseudo_iterators[level]; j<m3; j++){
		int candidate=G(parentv,j);
		bool found=false;
		for(int p=0; p<n; p++)
		  if(matching[p]==candidate){
		    found=true;
		    break;
		  }
		if(!found){
		  newv=candidate;
		  pseudo_iterators[level]=j+1;
		  break;
		}
	      }
	      if(newv!=-1){
		w=neww;
		v=newv;
		level++;
		break;
	      }
	      matching[Htraversal[level]]=-1;
	      level--;
	    }
	  }

	}
      }
      matches.resize0(nmatches);
    }

      
  private:
    
    void add_match(vector<int> matching){
      CNINE_ASSRT(matching.size()==n);
      std::sort(matching.begin(),matching.end());

      for(int i=0; i<nmatches; i++){
	    bool is_same=true;
	    for(int j=0; j<n; j++)
	      if(matches(i,j)!=matching[j]){
		is_same=false; 
		break;
	      }
	    if(is_same) return;
      }
      
      if(nmatches>matches.dim(0)-1)
	matches.resize0(std::max(5,2*(nmatches+1)));

      for(int j=0; j<n; j++)
	matches.set(nmatches,j,matching[j]);
      nmatches++;
    }

  };

}

#endif 
