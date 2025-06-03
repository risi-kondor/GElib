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

#ifndef _GatherMapProgramHelpers
#define _GatherMapProgramHelpers

#include "Cnine_base.hpp"
#include "GatherMapB.hpp"

namespace cnine{


  class GatherMapProgram;
  inline int makeGatherMapVar(GatherMapProgram&, const Gdims&);

  
  class GatherMapVar{
  public:

    friend class GatherMapProgram;
    friend class GatherMapProgramVariable;
    friend class GatherMapProgramInstruction;

    const int id;

    int roffset=0;
    int coffset=0;

    GatherMapVar(GatherMapProgram& prog, const Gdims& _dims):
      id(makeGatherMapVar(prog,_dims)){}

    GatherMapVar operator()(const int _roffset, const int _coffset){
      return GatherMapVar(id,_roffset,_coffset);
    }

  private:

    GatherMapVar(const int _id, const int _roffset=0, const int _coffset=0):
      id(_id), roffset(_roffset), coffset(_coffset){};

  };


  class GatherMapProgramVariable{
  public:

    int id;
    Gdims dims;


  public: // ---- Constructors --------------------------------------


    GatherMapProgramVariable(){}

    GatherMapProgramVariable(const int _id): 
      id(_id), dims({0,0}){}

    GatherMapProgramVariable(const int _id, const Gdims& _dims): 
      id(_id), dims(_dims){}


  public: // ---- Copying -------------------------------------------


    GatherMapProgramVariable(const GatherMapProgramVariable& x):
      id(x.id), dims(x.dims){}

    GatherMapProgramVariable& operator=(const GatherMapProgramVariable& x){
      id=x.id;
      dims=x.dims;
      return *this;
    }


  public: // ---- I/O -----------------------------------------------


    string repr() const{
      ostringstream oss;
      if(id==0) oss<<"input";
      if(id==1) oss<<"output";
      if(id>1) oss<<"v"<<to_string(id);
      return oss.str();
    }

    string str() const{
      ostringstream oss;
      if(id==0) oss<<"input";
      if(id==1) oss<<"output";
      if(id>1) oss<<"v"<<id;
      oss<<dims;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GatherMapProgramVariable& v){
      stream<<v.str(); return stream;}

  };

    

  class GatherMapProgramInstruction{
  public:

    int in;
    //int in_cdivider=1;
    //int in_coffset=0;
    //int in_ncols=0;

    int out;
    //int out_rdivider=0;
    //int out_roffset=0;
    //int out_nrows=0;

    shared_ptr<const GatherMapB> map;


  public: // ---- Constructors -------------------------------------------------------------------------------


    //Instruction(const int _out, const int _in, shared_ptr<const GatherMapB> _map):
    //in(_in), out(_out), map(_map){}

    GatherMapProgramInstruction(shared_ptr<const GatherMapB> _map, const GatherMapVar& _out, const GatherMapVar& _in):
      in(_in.id), //in_roffset(_in.roffset), in_coffset(_in.coffset), 
      out(_out.id), //out_roffset(_out.roffset), out_coffset(_out.coffset), 
      map(_map){}

    GatherMapProgramInstruction(const shared_ptr<GatherMapB>& _map, const int _out, const int _in):
      in(_in), out(_out), map(_map){}

    GatherMapProgramInstruction(const GatherMapB& _map, const int _out, const int _in):
      in(_in), out(_out), map(new GatherMapB(_map)){}

    GatherMapProgramInstruction(const GatherMapB* _map, const int _out, const int _in):
      in(_in), out(_out), map(to_share(_map)){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    GatherMapProgramInstruction(){}

    GatherMapProgramInstruction(const GatherMapProgramInstruction& x):
      GatherMapProgramInstruction(x.map,x.out,x.in){}

    GatherMapProgramInstruction& operator=(const GatherMapProgramInstruction& x){
      map=x.map;
      in=x.in;
      out=x.out;
      return *this;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    GatherMapProgramInstruction inv() const{
      return GatherMapProgramInstruction(map->inv_ptr(), // unnecessary copy here 
	[](const int x){if(x<2) return 1-x; else return x;}(in),
	[](const int x){if(x<2) return 1-x; else return x;}(out));
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string repr() const{
      return str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"V"<<to_string(out);
      //if(out_roffset>0 || out_coffset>0) oss<<"["<<out_roffset<<","<<out_coffset<<"]";
      oss<<"<-"<<"gather(V"<<to_string(in);
      //if(in_roffset>0 || in_coffset>0) oss<<"["<<in_roffset<<","<<in_coffset<<"]";
      oss<<")"<<endl;
      oss<<map->str(indent+"    ")<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GatherMapProgramInstruction& v){
      stream<<v.str(); return stream;}

  };


}

#endif 
