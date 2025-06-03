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

#ifndef _GatherMapProgram
#define _GatherMapProgram

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"
#include "GatherMapProgramHelpers.hpp"
#include "GatherRows.hpp"


namespace cnine{


  class GatherMapProgram{
  public: 

    typedef GatherMapProgramVariable Variable;
    typedef GatherMapProgramInstruction Instruction;


    vector<Variable> vars;
    vector<Instruction> instructions;
    int is_inverse=false;

    
  public: // ---- Constructors -------------------------------------------------------------------------------


    GatherMapProgram():
      GatherMapProgram(dims(1,1),dims(1,1)){
    }

    GatherMapProgram(const int  out_rows, const int in_rows){
      vars.push_back(Variable(0,Gdims(in_rows,0)));
      vars.push_back(Variable(1,Gdims(out_rows,0)));
    }

    GatherMapProgram(const Gdims& out_dims, const Gdims& in_dims){
      vars.push_back(Variable(0,in_dims));
      vars.push_back(Variable(1,out_dims));
    }

    GatherMapProgram(const Gdims& out_dims, const Gdims& in_dims, const GatherMapB& g){
      vars.push_back(Variable(0,in_dims));
      vars.push_back(Variable(1,out_dims));
      instructions.push_back(Instruction(g,1,0));
    }


    GatherMapProgram(const GatherMapB* g){
      vars.push_back(Variable(0));
      vars.push_back(Variable(1));
      instructions.push_back(Instruction(g,1,0));
    }

    GatherMapProgram(int out_rows, int in_rows, const GatherMapB* g){
      vars.push_back(Variable(0,Gdims(in_rows,1)));
      vars.push_back(Variable(1,Gdims(out_rows,1)));
      instructions.push_back(Instruction(g,1,0));
    }

    GatherMapProgram(const GatherMapB& g){
      vars.push_back(Variable(0));
      vars.push_back(Variable(1));
      instructions.push_back(Instruction(g,1,0));
    }



  public: // ---- Programming --------------------------------------------------------------------------------


    GatherMapVar input(const int i=0){
      return GatherMapVar(vars[0].id);
    }

    GatherMapVar output(const int i=0){
      return GatherMapVar(vars[1].id);
    }

    int add_var(const Gdims& _dims){
      vars.push_back(Variable(vars.size(),_dims));
      return vars.size()-1;
    }


    void add_map(const GatherMapB* map, const int out=1, const int in=0){
      instructions.push_back(Instruction(map,out,in));
    }

    void add_map(const GatherMapB& map, const int out=1, const int in=0){
      instructions.push_back(Instruction(map,out,in));
    }


    //[[deprecated]]
    //void gather(const GatherMapVar& out, const GatherMapVar& in, const GatherMapB* map){
    //gather(out,in,shared_ptr<const GatherMapB>(map));
    //}

    //[[deprecated]]
    //void gather(const GatherMapVar& out, const GatherMapVar& in, shared_ptr<const GatherMapB> map){
    //instructions.push_back(Instruction(map,out,in));
    //}
    
    GatherMapProgram inv() const{
      GatherMapProgram R;

      R.vars.resize(vars.size());
      R.vars[0]=vars[1];
      R.vars[1]=vars[0];
      for(int i=2; i<vars.size(); i++)
	R.vars[i]=vars[i];

      int ninst=instructions.size();
      R.instructions.resize(ninst);
      for(int i=0; i<ninst; i++){
	R.instructions[i]=instructions[ninst-1-i].inv();
      }

      R.is_inverse=!is_inverse;
      return R;
    }


  public: // ---- Execution ----------------------------------------------------------------------------------


    template<typename TYPE>
    void operator()(const TensorView<TYPE>& output, const TensorView<TYPE>& arg0){
      CNINE_ASSRT(output.get_dev()==arg0.get_dev());
      CNINE_ASSRT(arg0.ndims()==2);
      CNINE_ASSRT(output.ndims()==2);
      int nc=arg0.dim(1);
      int dev=output.get_dev();

      vector<Ltensor<TYPE>*> v(vars.size());
      v[0]=new Ltensor<TYPE>(arg0);
      v[1]=new Ltensor<TYPE>(output);

      for(int i=2; i<vars.size(); i++){
	int ncols=nc*vars[i].dims[1];
	if(is_inverse) ncols=output.dim(1)*vars[i].dims[1];
	v[i]=new Ltensor<TYPE>(Gdims(vars[i].dims[0],ncols),0,dev); // changed!
      }

      for(auto& p:instructions){
	CNINE_ASSRT(p.out<vars.size());
	CNINE_ASSRT(p.in<vars.size());
	GatherRows()(*v[p.out],*v[p.in],*p.map);
      }

      for(auto p:v) // changed!
	delete p;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------
    

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Variables:"<<endl;
      for(auto& p:vars)
	oss<<indent<<"  "<<p<<endl;
      oss<<indent<<"Instructions:"<<endl;
      for(auto& p:instructions)
	oss<<indent<<"  "<<p<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GatherMapProgram& v){
      stream<<v.str(); return stream;}
    

  };


  inline int makeGatherMapVar(GatherMapProgram& p, const Gdims& dims){
    return p.add_var(dims);
  }


}

#endif 
