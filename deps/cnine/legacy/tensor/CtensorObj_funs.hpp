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


#ifndef _CnineCtensorFunctions
#define _CnineCtensorFunctions

#include "Cnine_base.hpp"
#include "ExprTemplates.hpp"
#include "CscalarObj.hpp"
#include "CtensorObj.hpp"
//#include "ArrayOf.hpp"


namespace cnine{



  // -------------------------------------------------------------------------------------------------------- 


  CscalarObj norm2(const CtensorObj& x){
    CscalarObj r(x.get_nbu(),fill::zero);
    x.add_norm2_into(r);
    return r;
  }

  CscalarObj inp(const CtensorObj& x, const CtensorObj& y){
    CscalarObj r(x.get_nbu(),fill::zero);
    x.add_inp_into(r,y);
    return r;
  }

  CtensorObj ReLU(const CtensorObj& x, const float c=0){
    CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero);
    R.add_ReLU(x,c);
    return R;
  }



  // ---- Scalar Multiplication -----------------------------------------------------------------------------

  
  class Ctensor_Cscalar_prod_expr: public Printable{
  public:
    const CscalarObj& c;
    const CtensorObj& x;
    Ctensor_Cscalar_prod_expr(const CscalarObj& _c, const CtensorObj& _x):
      c(_c), x(_x){}
    operator CtensorObj() const{
      CtensorObj R(x.get_dims(),x.get_nbu(),fill::zero,x.get_dev());
      R.add(x,c);
      return R;
    }
    string str(const string indent="") const{
      return CtensorObj(*this).str(indent);
    }
  };

  
  Ctensor_Cscalar_prod_expr operator*(const CscalarObj& c, const CtensorObj& x){
    return Ctensor_Cscalar_prod_expr(c,x); 
  }

  Ctensor_Cscalar_prod_expr operator*(const CtensorObj& x, const CscalarObj& c){
    return Ctensor_Cscalar_prod_expr(c,x); 
  }
  

  CtensorObj& operator+=(CtensorObj& R, const Ctensor_Cscalar_prod_expr& expr){
    R.add(expr.x,expr.c);
    return R;
  }
  

  // ---- Matrix multiplication -----------------------------------------------------------------------------


  class Ctensor_Mprod_expr: public Printable{
  public:
    const CtensorObj& x;
    const CtensorObj& y;
    Ctensor_Mprod_expr(const CtensorObj& _x, const CtensorObj& _y):
      x(_x), y(_y){}
    operator CtensorObj() const{
      CtensorObj R(x.get_dims().Mprod(y.get_dims()),x.get_nbu(),fill::zero,x.get_dev());
      R.add_mprod(x,y);
      return R;
    }
    string str(const string indent="") const{
      return CtensorObj(*this).str(indent);
    }
  };
  
  Ctensor_Mprod_expr operator*(const CtensorObj& x, const CtensorObj& y){
    return Ctensor_Mprod_expr(x,y); 
  }

  CtensorObj& operator+=(CtensorObj& R, const Ctensor_Mprod_expr& expr){
    R.add_mprod(expr.x,expr.y);
    return R;
  }


  CtensorObj operator*(const Transpose<CtensorObj>& x, const CtensorObj& y){
    int I=x.obj.get_dims().combined(1,x.obj.get_dims().k());
    int J=y.get_dims().combined(1,y.get_dims().k());
    CtensorObj R(dims(I,J),fill::zero);
    R.add_mprod_TA(x.obj,y);
    return R;
  }

  
  // ---- Other functions ----------------------------------------------------------------------------------

  /*
    template<typename... Args>
    CtensorObj stack(const int ix, const CtensorObj& x, Args...args){
    return CtensorObj(fill::stack,ix,x,args...);
    }
    
    template<typename... Args>
    CtensorObj cat(const int ix, const CtensorObj& x, Args...args){
    return CtensorObj(fill::cat,ix,x,args...);
    }
  */




}

#endif 
  //Broadcast<CtensorObj> broadcast(const CtensorObj& x){
  //return Broadcast<CtensorObj>(x);
  //}

  //Scatter<CtensorObj> scatter(const CtensorObj& x){
  //return Scatter<CtensorObj>(x);
  //}

  /*
  Transpose<CtensorObj> transp(const CtensorObj& x){
    return Transpose<CtensorObj>(x);
  }

  Conjugate<CtensorObj> conj(const CtensorObj& x){
    return Conjugate<CtensorObj>(x);
  }

  Hermitian<CtensorObj> herm(const CtensorObj& x){
    return Hermitian<CtensorObj>(x);
  }
  */
