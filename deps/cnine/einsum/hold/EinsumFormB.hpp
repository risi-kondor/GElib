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


#ifndef _CnineEinsumFormB
#define _CnineEinsumFormB

#include "TensorView.hpp"
#include "EinsumParser.hpp"
#include "einsum_node.hpp"
#include "multivec.hpp"

namespace cnine{


  class EinsumFormB{
  public:

    vector<ix_entry> summations;
    vector<ix_entry> transfers;
    vector<ix_entry> contractions;

    vector<vector<vector<int> > > map_to_dims;
    vector<shared_ptr<input_node> > input_nodes;


    EinsumFormB(const string str){
      EinsumParser parser(str);
      vector<ix_entry> ix_entries=parser.ix_entries;
      input_nodes=parser.input_nodes;
      map_to_dims=parser.map_to_dims;

      for(auto& p:ix_entries){
	if(p.size()==1){
	  summations.push_back(p);
	  continue;
	}
	if(p[0].first==0) transfers.push_back(p);
	else contractions.push_back(p);
      }

    }

  };

}
#endif 
