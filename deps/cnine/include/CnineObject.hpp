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


#ifndef _CnineObject
#define _CnineObject

#include "Cnine_base.hpp"


namespace cnine{


  class CnineObject{
  public:

    CnineObject(){}

    virtual ~CnineObject(){}

    // virtual CnineObject& operator=(const CnineObject& x){CNINE_UNIMPL(); return *this;}

    //virtual CnineObject* clone() const=0;
    //virtual CnineObject* spawn(const fill_zero& fill) const{CNINE_UNIMPL(); return nullptr;}
    virtual CnineObject* spawn_zero() const{CNINE_UNIMPL(); return nullptr;}

    const CnineObject* ptr() const {return this;}
      
    virtual void clear(){set_zero();}; 
    virtual void set_zero()=0; 
    //virtual void add(const CnineObject& x, const float c){FCG_UNIMPL();}
    //virtual void subtract(const CnineObject& x){CNINE_UNIMPL();}
    //virtual void subtract(const CnineObject& x, const float c){FCG_UNIMPL();}

    virtual string classname() const=0; 
    virtual string describe() const {return "";} 
    virtual string short_str() const {return "";} 
    virtual string str(const string indent="") const=0; 

    //virtual void adam_update(CnineObject& mt, CnineObject& vt, 
    //const float beta1, const float beta2, const float alpha, const float epsilon) {FCG_UNIMPL();}
    //virtual void adagrad_update(CnineObject& G, const float eta, const float epsilon) {FCG_UNIMPL();}

    friend ostream& operator<<(ostream& stream, const CnineObject& x){
      stream<<x.str(); return stream;}

  };

  //class ArithmeticObj{};


  template<typename TYPE>
  inline TYPE& assume(CnineObject* x){
    if(!dynamic_cast<TYPE*>(x)){
      if(!x) cerr<<"cnine error: CnineObject does not exist."<<endl;
      else {TYPE dummy; cerr<<"cnine error: CnineObject is of type "<<x->classname()<<" instead of "<<dummy.classname()<<"."<<endl;}
    }
    assert(dynamic_cast<TYPE*>(x));
    return *static_cast<TYPE*>(x);
  }

  template<typename TYPE>
  inline TYPE& assume(CnineObject* x, const char* s){
    if(!dynamic_cast<TYPE*>(x)){
      cerr<<"In function "<<s<<endl;
      if(!x) cerr<<"cnine error: CnineObject does not exist."<<endl;
      else {TYPE dummy; cerr<<"cnine error: CnineObject is of type "<<x->classname()<<" instead of "<<dummy.classname()<<"."<<endl;}
    }
    assert(dynamic_cast<TYPE*>(x));
    return *static_cast<TYPE*>(x);
  }

  template<typename TYPE>
  inline const TYPE& assume(const CnineObject* x){
    if(!dynamic_cast<const TYPE*>(x)){
      TYPE dummy; cerr<<"Error: object is of type "<<x->classname()<<" instead of "<<dummy.classname()<<"."<<endl;}
    assert(dynamic_cast<const TYPE*>(x));
    return *static_cast<const TYPE*>(x);
  }

  template<typename TYPE>
  inline const TYPE& assume(const CnineObject* x, const char* s){
    if(!dynamic_cast<const TYPE*>(x)){
      cerr<<"In function "<<s<<endl; TYPE dummy;
      cerr<<"Error: object is of type "<<x->classname()<<" instead of "<<dummy.classname()<<"."<<endl;
    }
    assert(dynamic_cast<const TYPE*>(x));
    return *static_cast<const TYPE*>(x);
  }


  template<typename TYPE>
  inline TYPE& assume(CnineObject& x){
    //if(!dynamic_cast<TYPE*>(&x)) printf("In function \"%s\" \n",__PRETTY_FUNCTION__);
    assert(dynamic_cast<TYPE*>(&x));
    return static_cast<TYPE&>(x);
  }

  template<typename TYPE>
  inline TYPE& assume(CnineObject& x, const char* s){
    if(!dynamic_cast<TYPE*>(&x)){cerr<<"In function "<<s<<endl;}
    assert(dynamic_cast<TYPE*>(&x));
    return static_cast<TYPE&>(x);
  }

  template<typename TYPE>
  inline const TYPE& assume(const CnineObject& x){
    //if(!dynamic_cast<const TYPE*>(&x)) printf("In function \"%s\" \n",__PRETTY_FUNCTION__);
    assert(dynamic_cast<const TYPE*>(&x));
    return static_cast<const TYPE&>(x);
  }

  template<typename TYPE>
  inline const TYPE& assume(const CnineObject& x, const char* s){
    if(!dynamic_cast<const TYPE*>(&x)){cerr<<"In function "<<s<<endl;}
    assert(dynamic_cast<const TYPE*>(&x));
    return static_cast<const TYPE&>(x);
  }

  template<typename TYPE>
  inline bool istype(const CnineObject* x){
    return dynamic_cast<const TYPE*>(x)!=nullptr;
  }

  

}


#endif
