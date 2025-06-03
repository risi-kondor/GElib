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

#ifndef _TensorProgram
#define _TensorProgram

#include "Cnine_base.hpp"
#include "TensorProgramHelpers.hpp"
#include "GatherRows.hpp"


namespace cnine{

  class GatherMapB;

  template<typename OPERATION, typename MAP>
  class TensorProgramPack; 



  template<typename OPERATION, typename MAP>
  class TensorProgram{
  public: 

    typedef TensorProgramVariable VAR;
    typedef TensorProgramInstruction<MAP> INSTR;

    int inputvar=0;
    int outputvar=1;
    bool nc_from_output=false;

    vector<VAR> vars;
    vector<INSTR> instructions;


  public: // ---- Constructors -------------------------------------------------------------------------------


    TensorProgram(const Gdims& out_dims, const Gdims& in_dims){
      vars.push_back(VAR(in_dims));
      vars.push_back(VAR(out_dims));
    }

    TensorProgram(const Gdims& out_dims, const Gdims& in_dims, const MAP& g){
      vars.push_back(VAR(in_dims));
      vars.push_back(VAR(out_dims));
      instructions.push_back(INSTR(g,1,0));
    }

    TensorProgram(const Gdims& out_dims, const Gdims& in_dims, MAP* g){
      vars.push_back(VAR(in_dims));
      vars.push_back(VAR(out_dims));
      instructions.push_back(INSTR(g,1,0));
    }


  public: // ---- Programming --------------------------------------------------------------------------------


    int add_var(const Gdims& _dims){
      vars.push_back(VAR(_dims));
      return vars.size()-1;
    }

    void add_map(MAP* map, const int out=1, const int in=0){
      instructions.push_back(INSTR(map,out,in));
    }

    void add_map(const MAP& map, const int out=1, const int in=0){
      instructions.push_back(INSTR(map,out,in));
    }


  public: // ---- Operations -------------------------------------------------------------------------------


    template<typename TYPE>
    void operator()(const Ltensor<TYPE>& output, const Ltensor<TYPE>& input){

      int dev=input.get_dev();
      CNINE_ASSRT(output.get_dev()==input.get_dev());
      CNINE_ASSRT(inputvar<vars.size());
      CNINE_ASSRT(outputvar<vars.size());

      int nc;
      if(nc_from_output) 
	nc=output.get_dims().back()/vars[outputvar].dims.back();
      else 
      	nc=input.get_dims().back()/vars[inputvar].dims.back();

      CNINE_ASSRT(input.ndims()==vars[inputvar].dims.size());
      CNINE_ASSRT(input.get_dims()==scale_last(vars[inputvar].dims,nc));

      CNINE_ASSRT(output.ndims()==vars[outputvar].dims.size());
      CNINE_ASSRT(output.get_dims()==scale_last(vars[outputvar].dims,nc));

      vector<Ltensor<TYPE>*> v(vars.size());
      v[inputvar]=new Ltensor<TYPE>(input);
      v[outputvar]=new Ltensor<TYPE>(output);

      for(int i=0; i<vars.size(); i++){
	if(i==inputvar) v[i]=new Ltensor<TYPE>(input);
	if(i==outputvar) v[i]=new Ltensor<TYPE>(output);
	if(i!=inputvar && i!=outputvar)
	  v[i]=new Ltensor<TYPE>(scale_last(vars[i].dims,nc),0,dev);
      }

      for(auto& p:instructions){
	CNINE_ASSRT(p.out<vars.size());
	CNINE_ASSRT(p.in<vars.size());
	OPERATION()(*v[p.out],*v[p.in],*p.map);
      }

      for(auto p:v) 
	delete p;
    }


    TensorProgram<OPERATION,MAP> inv() const{
      TensorProgram<OPERATION,MAP> R(vars[inputvar].dims,vars[outputvar].dims);
      R.vars=vars;
      R.inputvar=outputvar;
      R.outputvar=inputvar;
      R.nc_from_output=!nc_from_output;

      int n=instructions.size();
      R.instructions.resize(n);
      for(int i=0; i<n; i++)
	R.instructions[i]=instructions[n-1-i].inv();

      return R;
    }


    static TensorProgram<OPERATION,GatherMapPack> fuse0(const TensorProgramPack<OPERATION,GatherMapB>& x){
      int N=x.size();
      CNINE_ASSRT(N>0);
      TensorProgram<OPERATION,GatherMapPack> R;

      R.inputvar=x[0].inputvar;
      R.outputvar=x[0].outputvar;
      R.nc_from_output=x[0].nc_from_output;

      int nvars=x[0].vars.size();
      int ninst=x[0].instructions.size();

      for(int j=0; j<N; j++){
	CNINE_ASSRT(x[j].inputvar==R.inputvar);
	CNINE_ASSRT(x[j].outputvar==R.outputvar);
	CNINE_ASSRT(x[j].nc_from_output==R.nc_from_output);
	CNINE_ASSRT(x[j].vars.size()==nvars);
	CNINE_ASSRT(x[j].instructions.size()==ninst);
      }

      for(int i=0; i<nvars; i++){
	Gdims D=x[0].vars[i].dims.chunk(1);
	int t=0;
	for(int j=0; j<N; j++){
	  CNINE_ASSRT(x[j].vars[i].dims.chunk(1)==D);
	  t+=x[j].vars[i].dims[0];
	}
	R.vars.push_back(VAR(Gdims(Gdims(t),D)));
      }

      for(int i=0; i<ninst; i++){
	int in=x[0].instructions[i].in;
	int out=x[0].instructions[i].out;
	vector<shared_ptr<GatherMapB> > v;
	for(int j=0; j<N; j++){
	  auto& inst=x[j].instruction[i];
	  CNINE_ASSRT(inst.in==in);
	  CNINE_ASSRT(inst.out==out);
	  v.push_back(inst.map);
	}
	R.instructions.push_back(TensorProgramInstruction<GatherMapPack>(GatherMapPack(v),out,in));
      }
      
      return R;
    }


  private: // ------------------------------------------------------------------------------------------------


    Gdims scale_last(const Gdims& x, const int nc){
      Gdims R(x);
      R.set_back(R.back()*nc);
      return R;
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

    friend ostream& operator<<(ostream& stream, const TensorProgram& v){
      stream<<v.str(); return stream;}


  };



  template<typename OPERATION, typename MAP>
  class TensorProgramPack: public shared_object_pack<TensorProgram<OPERATION,MAP> >{

    template<typename TYPE>
    void operator()(const Ltensor<TYPE>& output, const Ltensor<TYPE>& input){

    }

    

  };


}

#endif 
    //int nrows() const{
    //return dims[0];
    //}

    //int ncols() const{
    //return dims[1];
    //}

