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

#ifndef _TensorProgramHelpers
#define _TensorProgramHelpers

#include "Cnine_base.hpp"


namespace cnine{


  class TensorProgramVariable{
  public:

    Gdims dims;

    TensorProgramVariable(){}

    TensorProgramVariable(const int _nrows, const int _ncols):
      dims(_nrows,_ncols){}

    TensorProgramVariable(const Gdims& _dims):
      dims(_dims){
    }

    string str() const{
      ostringstream oss;
      oss<<"v";//<<id;
      oss<<dims;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorProgramVariable& v){
      stream<<v.str(); return stream;}

  };



  template<typename MAP>
  class TensorProgramInstruction{
  public:

    int in;
    int out;
    shared_ptr<MAP> map;

    TensorProgramInstruction(){}

    TensorProgramInstruction(const shared_ptr<MAP>& _map, const int _out, const int _in):
      in(_in), out(_out), map(_map){};

    TensorProgramInstruction(MAP* _map, const int _out, const int _in):
      in(_in), out(_out), map(shared_ptr<MAP>(_map)){};

    TensorProgramInstruction(const MAP& _map, const int _out, const int _in):
      in(_in), out(_out), map(shared_ptr<MAP>(new MAP(_map))){};

    TensorProgramInstruction inv() const{
      return TensorProgramInstruction(map->inv(),in,out);
    }

    string repr() const{
      return str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"V"<<to_string(out);
      oss<<"<-"<<"OP(V"<<to_string(in);
      oss<<")"<<endl;
      oss<<map->str(indent+"    ")<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const TensorProgramInstruction& v){
      stream<<v.str(); return stream;}

  };

}

#endif 
