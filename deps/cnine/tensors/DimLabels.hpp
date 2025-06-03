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


#ifndef _CnineDimLabels
#define _CnineDimLabels

#include "Cnine_base.hpp"
#include "Gdims.hpp"


namespace cnine{

  class DimLabels{
  public:

    bool _batched=false;
    int _narray=0;

    DimLabels(){}

    DimLabels(const bool b):
      _batched(b){}

    DimLabels(const bool b, const int na):
      _batched(b), _narray(na){}


  public: // ---- Copying -----------------------------------------------------------------------------------


    DimLabels copy() const{
      return *this;
    }


  public: // ---- Batched -----------------------------------------------------------------------------------


    int nbatch(const Gdims& dims) const{
      if(_batched) return dims[0];
      else return 0;
    }

    DimLabels& set_batched(const bool x){
      _batched=x;
      return *this;
    }


  public: // ---- Grid --------------------------------------------------------------------------------------


    Gdims gdims(const Gdims& dims) const{
      return dims.chunk(_batched,_narray);
    }

    GstridesB gstrides(const GstridesB& x) const{
      return x.chunk(_batched,_narray);
    }

    DimLabels& set_ngrid(const int x){
      _narray=x;
      return *this;
    }


  public: // ---- Cells --------------------------------------------------------------------------------------


    Gdims cdims(const Gdims& x) const{
      return x.chunk(_batched+_narray);
    }

    GstridesB cstrides(const GstridesB& x) const{
      return x.chunk(_batched+_narray);
    }


  public: // ---- Batched cells ------------------------------------------------------------------------------


    Gdims bcdims(const Gdims& x) const{
      if(!_batched) return x.chunk(_narray);
      return x.chunk(1+_narray).prepend(x[0]);
    }

    GstridesB bcstrides(const GstridesB& x) const{
      if(!_batched) return x.chunk(_narray);
      return x.chunk(1+_narray).prepend(x[0]);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "DimLabels";
    }

    string str(const Gdims& dims) const{
      ostringstream oss;
      
      oss<<"[";
      if(_batched) oss<<"nbatch="<<nbatch(dims)<<",";
      if(_narray>0) oss<<"blocks="<<gdims(dims)<<",";
      oss<<"dims="<<cdims(dims)<<" ";
      oss<<"\b]";
      return oss.str();
    }

  };

}

#endif

