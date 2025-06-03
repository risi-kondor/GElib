/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _Cnine_mtensor_constructor_ops
#define _Cnine_mtensor_constructor_ops

#include "Coperator.hpp"
#include "Tensor.hpp"


namespace cnine{

  typedef Cengine::Coperator Coperator;
  typedef Cengine::Cnode Cnode;


  template<typename TYPE>
  class new_mtensor_op: public Coperator{
  public:

    Gdims dims;
    int dev;

    new_mtensor_op(const Gdims& _dims, const int _dev=0):
      dims(_dims), dev(_dev){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(dims,fill::raw,dev);
    }

    string str() const{
      return "mtensor"+dims.str();
    }

  };


  template<typename TYPE>
  class new_mtensor_zero_op: public Coperator{
  public:

    Gdims dims;
    int dev;

    new_mtensor_zero_op(const Gdims& _dims, const int _dev=0):
      dims(_dims), dev(_dev){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(dims,fill::zero,dev);
    }

    string str() const{
      return "mtensor_zero"+dims.str();
    }

  };


  template<typename TYPE>
  class new_mtensor_ones_op: public Coperator{
  public:

    Gdims dims;
    int dev;

    new_mtensor_ones_op(const Gdims& _dims, const int _dev=0):
      dims(_dims), dev(_dev){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(dims,fill_constant(1),dev);
    }

    string str() const{
      return "mtensor_ones"+dims.str();
    }

  };


  template<typename TYPE>
  class new_mtensor_identity_op: public Coperator{
  public:

    Gdims dims;
    int dev;

    new_mtensor_identity_op(const Gdims& _dims, const int _dev=0):
      dims(_dims), dev(_dev){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(dims,fill::identity,dev);
    }

    string str() const{
      return "mtensor_identity"+dims.str();
    }

  };


  template<typename TYPE>
  class new_mtensor_sequential_op: public Coperator{
  public:

    Gdims dims;
    int dev;

    new_mtensor_sequential_op(const Gdims& _dims, const int _dev=0):
      dims(_dims),dev(_dev){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(dims,fill::sequential,dev);
    }

    string str() const{
      return "mtensor_sequential"+dims.str();
    }

  };


  template<typename TYPE>
  class new_mtensor_gaussian_op: public Coperator{
  public:

    Gdims dims;
    int dev;

    new_mtensor_gaussian_op(const Gdims& _dims, const int _dev=0):
      dims(_dims), dev(_dev){
    }

    //new_mtensor_gaussian_op(const Gdims& _dims, const float _c, const int _dev=0):
    //dims(_dims), c(_c), dev(_dev){
    //}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(dims,fill::gaussian,dev);
    }

    string str() const{
      return "mtensor_gaussian"+dims.str();
    }
    
  };


  /*
  class new_mtensor_from_gtensor_op: public Coperator{
  public:

    Gtensor<complex<float> > x;
    int nbu;
    int dev;

    new_mtensor_from_gtensor_op(const Gtensor<complex<float> >& _x, const int _nbu=-1, const int _dev=0):
      x(_x,nowarn), nbu(_nbu), dev(_dev){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(x,dev);
    }

    string str() const{
      return "mtensor()";
    }

  };
  */


  template<typename TYPE>
  class mtensor_copy_op: public Coperator{
  public:

    mtensor_copy_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(asTensor<TYPE>(inputs[0],__PRETTY_FUNCTION__));
    }

    string str() const{
      return "mtensor_copy("+inputs[0]->ident()+")";
    }
    
  };


  /*
    template<typename TYPE>
  class new_mtensor_fn2_op: public Coperator{
  public:

    Gdims dims;
    int nbu;
    int dev;
    std::function<complex<float>(const int, const int)> fn; 

    new_mtensor_fn2_op(const Gdims& _dims, const int _ 
      function<TYPE(const int, const int)> _fn, const int _dev=0):
      dims(_dims), nbu(_nbu), dev(_dev), fn(_fn){
    }

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(dims,fn,dev);
    }

    string str() const{
      return "mtensor_fn2"+dims.str();
    }

  };
  */


  /*
  template<typename TYPE>
  class mtensor_apply_op: public Coperator{
  public:

    std::function<TYPE(const complex<float>)> fn; 

    mtensor_apply_op(Cnode* x, std::function<complex<float>(const complex<float>)> _fn):
      Coperator(x), fn(_fn){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(MTENSORB(inputs[0]),fn);
    }
    
    string str() const{
      return "mtensor_apply"+inp_str();
    }

  };
  */


  /*
  template<typename TYPE>
  class mtensor_apply2_op: public Coperator{
  public:

    std::function<TYPE(const int, const int, const complex<float>)> fn; 

    mtensor_apply2_op(Cnode* x, std::function<complex<float>(const int, const int, const complex<float>)> _fn):
      Coperator(x), fn(_fn){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=new Tensor<TYPE>(MTENSORB(inputs[0]),fn);
    }
    
    string str() const{
      return "mtensor_apply"+inp_str();
    }

  };
  */



}

#endif 
