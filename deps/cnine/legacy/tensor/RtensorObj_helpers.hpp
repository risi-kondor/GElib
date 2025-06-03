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


#ifndef _RtensorObj_helpers
#define _RtensorObj_helpers


namespace cnine{

  class RtensorObj;

  class RtensorObj_element: public RscalarObjExpr{
  private:

    friend class RtensorObj;

    RtensorObj& obj;
    Gindex ix; // improve this

    RtensorObj_element(RtensorObj& _obj, const Gindex& _ix): 
      obj(_obj), ix(_ix){}

    RtensorObj_element(const RtensorObj_element& x): 
      obj(x.obj),ix(x.ix){};

  public:

    RtensorObj_element& operator=(const RtensorObj_element& x){
      operator=(x.operator RscalarObj());
      return *this;
    }

  public:

    operator RscalarObj() const;
    RtensorObj_element& operator=(const RscalarObj& x); 

  public:

    float get_value() const;
    RtensorObj_element& set_value(const float x);

  public: // shorthands

    float value() const{return get_value();};
    //operator complex<float>() const{return get_value();}; leads to ambiguity 

    RtensorObj_element& set(const float x) {set_value(x); return *this;}
    RtensorObj_element& operator=(const float x) {set_value(x); return *this;}

  public:

    string str(string indent="") const{
      float c=get_value();;
      return indent+to_string(c);
    }
  };


  class ConstRtensorObj_element: public RscalarObjExpr{
  public:
    friend class RtensorObj;
    const RtensorObj& obj;
    Gindex ix; // improve this
    ConstRtensorObj_element(const RtensorObj& _obj, const Gindex& _ix): 
      obj(_obj), ix(_ix){}
  private:
    ConstRtensorObj_element(const ConstRtensorObj_element& x):obj(x.obj),ix(x.ix){};
    ConstRtensorObj_element& operator=(const ConstRtensorObj_element& x)=delete;
  public:
    operator RscalarObj() const;
    operator float() const;
    float value() const;
    float get_value() const{return value();};
  public:
    string str(string indent="") const{
      float c=get_value();
      return indent+to_string(c);
    }
  };


}

#endif 
