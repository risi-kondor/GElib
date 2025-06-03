/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineTensorBase
#define _CnineTensorBase

#include "Cnine_base.hpp"


namespace cnine{

  class TensorBase{
  public:


    virtual ~TensorBase(){}


  public: // ---- Getters -----------------------------------------------------------------------------------


    template<typename TYPE>
    TYPE get(const int i0){
    }

  public: // ---- I/O ---------------------------------------------------------------------------------------


    virtual string str(const string indent="") const=0;


  };


}


#endif 


    /*
  public: // ---- int getters --------------------------------------------------------------------------------


    int get_int(const int i0)=0;
    int get_int(const int i0, const int i1)=0;
    int get_int(const int i0, const int i1, const int i2)=0;

    int set(const int i0, const int v)=0;
    int set(const int i0, const int i1, const int v)=0;
    int set(const int i0, const int i1, const int i2, const int v)=0;


  public: // ---- float getters ------------------------------------------------------------------------------


    int get_float(const int i0)=0;
    int get_float(const int i0, const int i1)=0;
    int get_float(const int i0, const int i1, const int i2)=0;

    int set(const int i0, const float v)=0;
    int set(const int i0, const int i1, const float v)=0;
    int set(const int i0, const int i1, const int i2, const float v)=0;


  public: // ---- float getters ------------------------------------------------------------------------------


    int get_float(const int i0)=0;
    int get_float(const int i0, const int i1)=0;
    int get_float(const int i0, const int i1, const int i2)=0;

    int set(const int i0, const float v)=0;
    int set(const int i0, const int i1, const float v)=0;
    int set(const int i0, const int i1, const int i2, const float v)=0;
    */
