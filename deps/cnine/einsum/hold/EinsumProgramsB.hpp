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


#ifndef _CnineEinsumProgramsB
#define _CnineEinsumProgramsB

#include "TensorView.hpp"
#include "EinsumFormB.hpp"
#include "EinsumProgramB.hpp"
#include "LatexDoc.hpp"


namespace cnine{


  class EinsumProgramsB{
  public:

    vector<EinsumProgramB> programs;
    vector<shared_ptr<input_node> > arg_nodes;
    vector<ix_entry> contractions;

    EinsumProgramsB(const EinsumFormB& form){
      arg_nodes=form.arg_nodes;
      contractions=form.contractions;
      
      EinsumProgramB prg(arg_nodes);
      vector<int> remaining_contractions;
      for(int i=0; i<contractions.size(); i++)
	remaining_contractions.insert(i);
      build_programs(prg,form.contraction_ids);
    }


  public: // ---- Building all possible programs to express form ---------------------------------------------


    void build_programs(const EinsumProgram& prg, const vector<int>& remaining){
      if(remaining.size()==0){
	for(auto& x:programs)
	  if(prg.levelwise_equal(x)) return;
	programs.push_back(prg);
	return;
      }
      for(int i=0; i<remaining.size(); i++){
	//cout<<i<<endl;
	EinsumProgramB sub_prg(prg);
	sub_prg.add_contraction(contractions(remaining[i]));
	vector<int> sub_remaining(remaining);
	sub_remaining.erase(sub_remaining.begin()+i);
	build_programs(sub_prg,sub_remaining);
      }
    }


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
