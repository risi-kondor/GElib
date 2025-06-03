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


#ifndef _CnineEinsumPrograms
#define _CnineEinsumPrograms

#include "TensorView.hpp"
#include "EinsumFormN.hpp"
//#include "EinsumFormB.hpp"
#include "EinsumProgram.hpp"
#include "LatexDoc.hpp"


namespace cnine{


  class EinsumPrograms{
  public:

    vector<EinsumProgram> programs;
    vector<shared_ptr<einsum_node> > args;

    EinsumPrograms(const EinsumFormN& form){
      args=form.arg_nodes;
      EinsumProgram prg(args);
      set<int> remaining;
      //for(auto p:form.contraction_ids)
      //remaining.push_back(p);
      cout<<form.contraction_ids<<endl;
      build_programs(prg,form.contraction_ids);
    }

    EinsumPrograms(const EinsumFormB& form){
      args=form.arg_nodes;
      EinsumProgram prg(args);
      set<int> remaining;
      //for(auto p:form.contraction_ids)
      //remaining.push_back(p);
      cout<<form.contraction_ids<<endl;
      build_programs(prg,form.contraction_ids);
    }


  public: // ---- Building all possible programs to express form ---------------------------------------------


    void build_programs(const EinsumProgram& prg, const vector<int>& rem){
      if(rem.size()==0){
	for(auto& x:programs)
	  if(prg.levelwise_equal(x)) return;
	programs.push_back(prg);
	return;
      }
      for(int i=0; i<rem.size(); i++){
	//cout<<i<<endl;
	EinsumProgram sub_prg(prg);
	sub_prg.add_contraction(rem[i]);
	vector<int> sub_rem(rem);
	sub_rem.erase(sub_rem.begin()+i);
	build_programs(sub_prg,sub_rem);
      }
    }

    /*
    bool contains(const EinsumProgram& prg){
      for(auto& x:programs){
	for(auto& p:prg.levels){
	  auto& l=x.levels.find(p.first);
	  if(l==levels.end()) return false;
	  if(!std::includes(p.second.begin(),)
	}
      }
    }
    */


  public: // ---- I/O ----------------------------------------------------------------------------------------


    void latex(string filename="temp") const{
      ostringstream oss;
      for(auto& prg:programs)
	prg.latex(oss);
      LatexDoc doc(oss.str());
      ofstream ofs(filename+".tex");
      ofs<<doc;
      ofs.close();
    }
    
    string str(const string indent="") const{
      ostringstream oss;
      for(auto& p:programs)
	oss<<p.str(indent)<<endl;
      return oss.str();
    }


    friend ostream& operator<<(ostream& stream, const EinsumPrograms& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
