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


#ifndef _Cnine_mtensor_ops
#define _Cnine_mtensor_ops

#include "Coperators.hpp"
#include "Tensor.hpp"


namespace cnine{

  typedef Cengine::Coperator Coperator;
  typedef Cengine::Cnode Cnode;


  template<typename TYPE>
  inline Tensor<TYPE>& asTensor(Cengine::Cobject* x, const char* s){
    return Cengine::downcast<Tensor<TYPE> >(x,s);
  }

  template<typename TYPE>
  inline Tensor<TYPE>& asTensor(Cengine::Cnode* x, const char* s){
    return Cengine::downcast<Tensor<TYPE> >(x,s);
  }

#define MTENSOR(x) asTensor<TYPE>(x,__PRETTY_FUNCTION__) 



  // ---- Not in-place operators  ----------------------------------------------------------------------------


  template<typename TYPE>
  class mtensor_conj_op: public Coperator{
  public:

    mtensor_conj_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=asTensor<TYPE>(inputs[0],__PRETTY_FUNCTION__).conj();
    }

    string str() const{
      return "mtensor_conj"+inp_str();
    }

  };
  

  template<typename TYPE>
  class mtensor_transp_op: public Coperator{
  public:

    mtensor_transp_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=asTensor<TYPE>(inputs[0],__PRETTY_FUNCTION__).transp();
    }

    string str() const{
      return "mtensor_transp"+inp_str();
    }

  };
  

  template<typename TYPE>
  class mtensor_herm_op: public Coperator{
  public:

    mtensor_herm_op(Cnode* x):
      Coperator(x){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=asTensor<TYPE>(inputs[0],__PRETTY_FUNCTION__).herm();
    }

    string str() const{
      return "mtensor_herm"+inp_str();
    }

  };
  

  // ---- Normalization  -------------------------------------------------------------------------------------


  template<typename TYPE>
  class mtensor_add_col_norms_op: public Cengine::CumulativeOp2<Tensor<TYPE> ,Tensor<TYPE> >{
  public:

    using Cengine::CumulativeOp2<Tensor<TYPE>,Tensor<TYPE> >::CumulativeOp2;

    virtual void exec(Tensor<TYPE> & R, const Tensor<TYPE> & x){
      R.add_col_norms(x);
    }

    string str() const{
      return "mtensor_add_col_norms"; //+inp_str();
    }

  };


  template<typename TYPE>
  class mtensor_add_col_norms_back_op: 
    public Cengine::CumulativeOp4<Tensor<TYPE> ,Tensor<TYPE> ,Tensor<TYPE> ,Tensor<TYPE> >{
  public:

    using Cengine::CumulativeOp4<Tensor<TYPE>,Tensor<TYPE>,Tensor<TYPE>,Tensor<TYPE> >::CumulativeOp4;

    virtual void exec(Tensor<TYPE> & R, const Tensor<TYPE> & g, const Tensor<TYPE> & x, const Tensor<TYPE> & n){
      R.add_col_norms_back(g,x,n);
    }

    string str() const{
      return "mtensor_add_col_norms_back"; //+inp_str();
    }

  };


  template<typename TYPE>
  class mtensor_divide_cols_op: public Coperator{
  public:

    mtensor_divide_cols_op(Cnode* r, Cnode* n):
      Coperator(r,n){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=CTENSORB(inputs[0]).divide_cols(CTENSORB(inputs[1]));
    }

    string str() const{
      return "mtensor_divide_cols"; //+inp_str();
    }

  };


  template<typename TYPE>
  class mtensor_add_divide_cols_op: public Cengine::CumulativeOp3<Tensor<TYPE> ,Tensor<TYPE> ,Tensor<TYPE> >{
  public:

    using Cengine::CumulativeOp3<Tensor<TYPE>,Tensor<TYPE>,Tensor<TYPE> >::CumulativeOp3;

    virtual void exec(Tensor<TYPE> & R, const Tensor<TYPE> & x, const Tensor<TYPE> & n){
      R.add_divide_cols(x,n);
    }

    string str() const{
      return "mtensor_add_divide_cols"; //+inp_str();
    }

  };


  template<typename TYPE>
  class mtensor_add_divide_cols_back0_op: 
    public Cengine::CumulativeOp3<Tensor<TYPE> ,Tensor<TYPE> ,Tensor<TYPE> >{
  public:

    using Cengine::CumulativeOp3<Tensor<TYPE>,Tensor<TYPE>,Tensor<TYPE> >::CumulativeOp3;

    virtual void exec(Tensor<TYPE> & R, const Tensor<TYPE> & g, const Tensor<TYPE> & n){
      R.add_divide_cols_back0(g,n);
    }

    string str() const{
      return "mtensor_add_divide_cols_back0"; //+inp_str();
    }

  };


  template<typename TYPE>
  class mtensor_add_divide_cols_back1_op: 
    public Cengine::CumulativeOp4<Tensor<TYPE> ,Tensor<TYPE> ,Tensor<TYPE> ,Tensor<TYPE> >{
  public:

    using Cengine::CumulativeOp4<Tensor<TYPE> ,Tensor<TYPE> ,Tensor<TYPE> ,Tensor<TYPE> >::CumulativeOp4;

    virtual void exec(Tensor<TYPE> & R, const Tensor<TYPE> & g, const Tensor<TYPE> & x, const Tensor<TYPE> & n){
      R.add_divide_cols_back1(g,x,n);
    }

    string str() const{
      return "mtensor_add_divide_cols_back1"; //+inp_str();
    }

  };




  // ---- In-place operators  --------------------------------------------------------------------------------

  
  template<typename TYPE>
  class mtensor_zero_op: public Coperator, public Cengine::InPlaceOperator{ // DEPRECATED 
  public:

    mtensor_zero_op(Cnode* r):
      Coperator(r){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asTensor<TYPE>(owner,__PRETTY_FUNCTION__).zero();
    }

    string str() const{
      return "mtensor_zero"+inp_str();
    }

  };
  
  
  template<typename TYPE>
  class mtensor_set_zero_op: public Coperator, public Cengine::InPlaceOperator{
  public:

    mtensor_set_zero_op(Cnode* r):
      Coperator(r){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asTensor<TYPE>(owner,__PRETTY_FUNCTION__).zero();
    }

    string str() const{
      return "mtensor_zero"+inp_str();
    }

  };


  // ---- Access ---------------------------------------------------------------------------------------------


  /*
  template<typename TYPE>
  class mtensor_get_element_op: public Coperator{
  public:

    Gindex ix;

    mtensor_get_element_op(Cnode* x, const Gindex& _ix):
      Coperator(x), ix(_ix){}

    virtual void exec(){
      assert(!owner->obj);
      const Tensor<TYPE>& x=CTENSORB(inputs[0]);
      owner->obj=new CscalarB(x.nbu,x(ix),x.device);
    }

    string str() const{
      return "mtensor_get_element"+inp_str(ix);
    }

  };
  */

  /*
  template<typename TYPE>
  class mtensor_set_element_op: public Coperator, public Cengine::InPlaceOperator{
  public:

    Gindex ix;

    mtensor_set_element_op(Cnode* r, Cnode* x, const Gindex& _ix):
      Coperator(r,x), ix(_ix){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      asTensor<TYPE>(owner,__PRETTY_FUNCTION__).set(ix,asCscalarB(inputs[1],__PRETTY_FUNCTION__).val);
    }

    string str() const{
      return "mtensor_set_element"+inp_str(ix);
    }

  };
  */


  /*
  template<typename TYPE>
  class mtensor_set_chunk_op: public Coperator, public Cengine::InPlaceOperator{
  public:

    int ix;
    int offs;

    mtensor_set_chunk_op(Cnode* r, Cnode* x, const int _ix, const int _offs):
      Coperator(r,x), ix(_ix), offs(_offs){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CTENSORB(owner).set_chunk(CTENSORB(inputs[1]),ix,offs);
    }

    string str() const{
      return "mtensor_set_chunk"+inp_str(ix,offs);
    }

  };
  */


  template<typename TYPE>
  class mtensor_to_device_op: public Coperator, public Cengine::InPlaceOperator{
  public:

    int dev;

    mtensor_to_device_op(Cnode* r, const int _dev):
      Coperator(r), dev(_dev){}

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      MTENSOR(owner).move_to_device(dev);
    }

    string str() const{
      return "mtensor_to_device"+inp_str(dev);
    }

  };


  
}

#endif 
