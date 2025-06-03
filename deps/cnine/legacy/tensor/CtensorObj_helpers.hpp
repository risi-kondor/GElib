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


#ifndef _CtensorObj_helpers
#define _CtensorObj_helpers


namespace cnine{

  class CtensorObj;

  class CtensorObj_element: public CscalarObjExpr{
  private:

    friend class CtensorObj;

    CtensorObj& obj;
    Gindex ix; // improve this

    CtensorObj_element(CtensorObj& _obj, const Gindex& _ix): 
      obj(_obj), ix(_ix){}

    CtensorObj_element(const CtensorObj_element& x): 
      obj(x.obj),ix(x.ix){};

  public:

    CtensorObj_element& operator=(const CtensorObj_element& x){
      operator=(x.operator CscalarObj());
      return *this;
    }

  public:

    operator CscalarObj() const;
    CtensorObj_element& operator=(const CscalarObj& x); 

  public:

    complex<float> get_value() const;
    CtensorObj_element& set_value(const complex<float> x);

  public: // shorthands

    complex<float> value() const{return get_value();};
    //operator complex<float>() const{return get_value();}; leads to ambiguity 
    CtensorObj_element& set(const complex<float> x) {set_value(x); return *this;}
    CtensorObj_element& operator=(const complex<float> x) {set_value(x); return *this;}

  public:

    string str(string indent="") const{
      complex<float> c=get_value();;
      return indent+"("+to_string(std::real(c))+","+to_string(std::imag(c))+")";}

  };


  class ConstCtensorObj_element: public CscalarObjExpr{
  public:
    friend class CtensorObj;
    const CtensorObj& obj;
    Gindex ix; // improve this
    ConstCtensorObj_element(const CtensorObj& _obj, const Gindex& _ix): 
      obj(_obj), ix(_ix){}
  private:
    ConstCtensorObj_element(const ConstCtensorObj_element& x):obj(x.obj),ix(x.ix){};
    ConstCtensorObj_element& operator=(const ConstCtensorObj_element& x)=delete;
  public:
    operator CscalarObj() const;
    operator complex<float>() const;
    complex<float> value() const;
    complex<float> get_value() const{return value();};
  public:
    string str(string indent="") const{
      complex<float> c=get_value();
      return indent+"("+to_string(std::real(c))+","+to_string(std::imag(c))+")";}
  };




}

#endif 
