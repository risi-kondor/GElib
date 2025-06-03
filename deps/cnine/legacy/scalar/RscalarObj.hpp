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


#ifndef _cnineRscalarObj
#define _cnineRscalarObj

#include "CnineObject.hpp"
#include "RscalarA.hpp"

#ifdef _WITH_CENGINE
//#include "RscalarM.hpp"
#endif 

namespace cnine{

  class RscalarObj;

  class RscalarObjExpr{
  public:
    virtual operator RscalarObj() const=0;
  };


  class RscalarObj: public CNINE_RSCALAR_IMPL{
  public:

    using CNINE_RSCALAR_IMPL::CNINE_RSCALAR_IMPL; 


  public: // ---- Named constructors -------------------------------------------------------------------------


    static RscalarObj zero(){
      return RscalarObj(fill::zero);}
    static RscalarObj zero(const int nbd=-1){
      return RscalarObj(nbd,fill::zero);}

    static RscalarObj gaussian(){
      return RscalarObj(fill::gaussian);}
    static RscalarObj gaussian(const int nbd=-1){
      return RscalarObj(nbd,fill::gaussian);}


  public: // ---- Copying ------------------------------------------------------------------------------------


    RscalarObj(const RscalarObj& x):
      CNINE_RSCALAR_IMPL(x){}

    RscalarObj(RscalarObj&& x):
      CNINE_RSCALAR_IMPL(std::move(x)){}

    RscalarObj& operator=(const RscalarObj& x){
      CNINE_RSCALAR_IMPL::operator=(x);
      return *this;
    }

    RscalarObj& operator=(RscalarObj&& x){
      CNINE_RSCALAR_IMPL::operator=(std::move(x));
      return *this;
    }

    /*
    Dobject* clone() const{
      return new RscalarObj(*this);
    }

    Dobject* spawn(const fill_zero& fill) const{
      return new RscalarObj(nbu,fill::zero);
    }
    */

    CnineObject* spawn_zero() const{
      return new RscalarObj(nbu,fill::zero);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    RscalarObj(const CNINE_RSCALAR_IMPL& x):
      CNINE_RSCALAR_IMPL(x){};
      
    /*
    RscalarObj& operator=(const Dobject& _x){
      assert(dynamic_cast<const RscalarObj*>(&_x));
      auto& x=static_cast<const RscalarObj&>(_x);
      (*this)=x;
      return *this;
    }
    */


  public: // ---- Access -------------------------------------------------------------------------------------



  public: // ---- In-place operations ------------------------------------------------------------------------


    void clear(){
      set_zero();
    }


  public: // ---- Non-inplace operations ---------------------------------------------------------------------


    RscalarObj plus(const RscalarObj& x){
      RscalarObj R(*this);
      R.add(x);
      return R;
    }

    RscalarObj apply(std::function<float(const float)> fn){
      return RscalarObj(CNINE_RSCALAR_IMPL::apply(fn));
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


 

  public: // ---- Into operations -----------------------------------------------------------------------------


    void inp_into(RscalarObj& R, const RscalarObj& y) const{
      R.add_prod(*this,y);
    }

    void norm2_into(RscalarObj& R) const{
      R.add_prod(*this,*this);
    }


  public: // ---- In-place operators --------------------------------------------------------------------------


    RscalarObj& operator+=(const RscalarObj& y){
      add(y);
      return *this;
    }

    RscalarObj& operator-=(const RscalarObj& y){
      subtract(y);
      return *this;
    }


  // ---- Binary operators -----------------------------------------------------------------------------------


  RscalarObj operator+(const RscalarObj& y) const{
    RscalarObj R(*this);
    R.add(y);
    return R;
  }

  RscalarObj operator-(const RscalarObj& y) const{
    RscalarObj R(*this);
    R.subtract(y);
    return R;
  }

  RscalarObj operator*(const RscalarObj& y) const{
    RscalarObj R(fill::zero);
    R.add_prod(*this,y);
    return R;
  }

  RscalarObj operator/(const RscalarObj& y) const{
    RscalarObj R(fill::zero);
    R.add_div(*this,y);
    return R;
  }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string classname() const{
      return "cnine::RscalarObj";
    }

    string describe() const{
      return "Rscalar";
    } 

    string str(const string indent="") const{
      return CNINE_RSCALAR_IMPL::str(indent);
    }

    friend ostream& operator<<(ostream& stream, const RscalarObj& x){
      stream<<x.str(); return stream;}

  };

  
  inline RscalarObj& asRscalar(CnineObject* x){
    assert(x); 
    if(!dynamic_cast<RscalarObj*>(x))
      cerr<<"cnine error: object is of type "<<x->classname()<<" instead of RscalarObj."<<endl;
    assert(dynamic_cast<RscalarObj*>(x));
    return static_cast<RscalarObj&>(*x);
  }

  /*
  inline CNINE_RSCALAR_IMPL& asRscalar(CnineObject* x){
    assert(x); 
    if(!dynamic_cast<CNINE_RSCALAR_IMPL*>(x))
      cerr<<"cnine error: object is of type "<<x->classname()<<" instead of RscalarA."<<endl;
    assert(dynamic_cast<CNINE_RSCALAR_IMPL*>(x));
    return static_cast<CNINE_RSCALAR_IMPL&>(*x);
  }
  */

  /*
  inline CNINE_RSCALAR_IMPL& asRscalarA(Cnode* x){
    assert(x->obj);
    if(!dynamic_cast<CNINE_RSCALAR_IMPL*>(x->obj))
      cerr<<"cnine error: object is of type "<<x->obj->classname()<<" instead of RscalarA."<<endl;
    assert(dynamic_cast<CNINE_RSCALAR_IMPL*>(x->obj));
    return static_cast<CNINE_RSCALAR_IMPL&>(*x->obj);
  }
  */


}

#endif
