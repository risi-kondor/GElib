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

#ifndef _CnineFindPlantedSubgraphs
#define _CnineFindPlantedSubgraphs

#include "Cnine_base.hpp"
#include "sparse_graph.hpp"
#include "labeled_tree.hpp"
#include "labeled_forest.hpp"
#include "TensorView.hpp"
#include "flog.hpp"


namespace cnine{


  template<typename LABEL=int>
  class FindPlantedSubgraphs{
  public:

    typedef sparse_graph<int,int,LABEL> Graph;

    const Graph& G;
    const Graph& H;
    int n;
    vector<pair<int,int> > Htraversal;
    vector<int> assignment;
    labeled_forest<int> matches;


  public:


    FindPlantedSubgraphs(const Graph& _G, const Graph& _H):
      G(_G), H(_H), n(_H.getn()){
      labeled_tree<int> S=H.greedy_spanning_tree();
      Htraversal=S.indexed_depth_first_traversal();
      assignment=vector<int>(n,-1);

      for(int i=0; i<G.getn(); i++){
	labeled_tree<int>* T=new labeled_tree<int>(i);
	matches.push_back(T);
	if(!make_subtree(*T,0)){
	  delete T;
	  matches.pop_back();
	}
      }
    }

    int nmatches() const{
      int t=0;
      for(auto p:matches)
	p->for_each_maximal_path([&](const vector<int>& x){t++;});
      return t;
    }


    operator cnine::TensorView<int>(){
      int N=nmatches();
      cnine::TensorView<int> R(cnine::dims(N,n),0,0);
      int t=0;
      for(auto p:matches)
	p->for_each_maximal_path([&](const vector<int>& x){
	    for(int i=0; i<n; i++) R.set(t,i,x[i]);
	    t++;});
      return R;
    }


  private:


    bool make_subtree(labeled_tree<int>& node, const int m){

      CNINE_ASSRT(m<Htraversal.size());
      const int v=Htraversal[m].first;
      const int w=node.label;

      if(G.is_labeled() && H.is_labeled() && G.labels.row(w)!=H.labels.row(v)) return false;
      if(H.with_degrees() && (H.degrees(v)>=0) && (G.data[w].size()!=H.degrees(v))) return false;

      for(auto& p:const_cast<Graph&>(H).data[v]){ // improve syntax
	if(assignment[p.first]==-1) continue;
	if(p.second!=G(w,assignment[p.first])){
	  return false;
	}
      }


      for(auto& p:const_cast<Graph&>(G).data[w]){
	auto it=std::find(assignment.begin(),assignment.end(),p.first);
	if(it==assignment.end()) continue;
	if(p.second!=H(v,it-assignment.begin())){
	  return false;
	}
      }

      assignment[v]=w;
      //cout<<string(m,' ')<<"matched "<<v<<" to "<<w<<endl;
      if(m==n-1){
	node.label=-1;
	//cout<<string(m,' ')<<"assignment ";
	//for(auto p:assignment)cout<<p; cout<<endl;
	bool is_duplicate=matches.contains_rooted_path_consisting_of(assignment);
	node.label=w;
	assignment[v]=-1;
	//cout<<string(m,' ')<<"duplicate="<<is_duplicate<<endl;
	return !is_duplicate;
      }

      // try to match next vertex in Htraversal to each neighbor of newparent  
      const int newparent=assignment[Htraversal[Htraversal[m+1].second].first];
      //cout<<string(m,' ')<<"newparent="<<newparent<<endl;
      for(auto& w:G.neighbors(newparent)){
	if(std::find(assignment.begin(),assignment.end(),w)!=assignment.end()) continue;
	labeled_tree<int>* T=new labeled_tree<int>(w);
	node.push_back(T);
	if(!make_subtree(*T,m+1)){
	  delete T;
	  node.children.pop_back();
	}
      }

      assignment[v]=-1;
      return node.children.size()>0;
    }

  };


}

#endif
