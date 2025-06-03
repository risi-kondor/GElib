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


#ifndef _CnineEinsumN
#define _CnineEinsumN

#include "TensorView.hpp"
#include "EinsumFormN.hpp"
#include "EinsumPrograms.hpp"

namespace cnine{


  class EinsumN{
  public:

    EinsumFormN form;
    
    shared_ptr<EinsumPrograms> programs;

    TensorView<TYPE> r;
    vector<TensorView<TYPE> > args;

    EinsumN(const string str):
      form(str){
      programs=make_shared<EinsumPrograms>(form);
    }
    
    void add_einsum()(const TensorView<TYPE>& _r, const Args&... _args){
      r.reset(_r);
      args.clear();
      unroller(_args...);
      apply();
    }
     
    void unroller(const TensorView<TYPE>& x, const Args&... _args){
      args.push_back(x);
      unroller(_args...);
    }

    void unroller(){}
    

  public: // ------------------------------------------------------------------------------------------------


    void apply(){

      auto& arg_nodes=form.result_node;
      auto& arg_nodes=form.arg_nodes;
      int Narg=arg_nodes.size();
      CNINE_ASSERT(args.size()==Narg);
 
     vector<int> dimensions(form.id_tail,-1);

     CNINE_ASSRT(r.ndims()=result_node->ids.size());
     for(int j=0; j<result_node->ids.size(); j++){
       int& d=dimensions[result_node->ids[j]];
       if(d==-1) d=r.dim(j);
       else CNINE_ASSRT(d==r.dim(j));
     }

     for(int i=0; i<Narg; i++){
	CNINE_ASSRT(args[i].ndims()==arg_nodes[i]->ids.size());
	for(int j=0; j<arg_nodes[i]->ids.size(); j++){
	  int& d=dimensions[arg_nodes[i]->ids[j]];
	  if(d==-1) d=args[i].dim(j);
	  else CNINE_ASSRT(d==args[i].dim(j));
	}
      }
    }

  };

}

#endif 
