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


#ifndef _CnineEinsumJit
#define _CnineEinsumJit

#include "TensorView.hpp"
#include "EinsumForm.hpp"


namespace cnine{

  class EinsumJit{
  public:

    EinsumForm& form;
    vector<TensorView<TYPE> > args;
    string code;

    template<typename... Args> 
    EinsumJit(const EinsumForm& _form, const Args&... _args):
      form(_form){
      unroller(_args...);
      make();
    }
     
    template<typename... Args> 
    void unroller(const TensorView<TYPE>& x, const Args&... _args){
      args.push_back(x);
      unroller(_args...);
    }

    void unroller(){}
    
 
  public: // ------------------------------------------------------------------------------------------------


    void make(){
      int nargs=form.args.size();
      auto& transfer_indices=form.transfer_indices;
      auto& map_to_dims=form.map_to_dims;

      ostringstream oss;
      oss<<"#include<TensorView.hpp>\n";
      oss<<"using namespace cnine;\n";
      oss<<"\n";
      oss<<"void einsum(const TensorView<float>& r, vector<const TensorView<float> >& args){\n\n";
      
      //oss<<"roffs=0;\n";
      //string roffs_str="roffs=0";
      vector<string> offs_str;
      for(int i=0; i<nargs; i++)
	offs_str.push_back("offs"+to_string(i)+"=0");

      for(int i=0; i<form.transfer_indices.size(); i++){
	auto& ix_entry=transfer_indices[i];
	string t_str="t"+to_string(i);
	int d=args[0].dims[map_to_dims[0][ix_entry[0].second][0]];
	oss<<string(' ',2*i)<<"for(int "<<t_str<<"=0; "<<t_str<<"<"<<d<<"; "<<t_str<<"++){\n";

	for(auto& p:ix_entry){
	  int s=args[p.first].strides.combine(map_to_dims[p.first][p.second]);
	  offs_str[p.first]+="+"+t_str+"*"+to_string(s);
	}
      }

      for(int i=0; i<nargs; i++)
	oss<<offs_str[i]<<";\n";

      for(i=form.transfer_indices.size()-1; i>=0; i--){
	oss<<string(' ',2*i)<<"}\n";
      }

      oss<<"}\n";
      code=oss.str();
    }


  };

}

#endif 
