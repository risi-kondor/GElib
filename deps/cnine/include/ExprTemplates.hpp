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


#ifndef _CnineExprTemplates
#define _CnineExprTemplates


namespace cnine{


  template<typename OBJ>
  class Transpose{
  public:
    const OBJ& obj;
    explicit Transpose(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ>
  class Conjugate{
  public:
    const OBJ& obj;
    explicit Conjugate(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ>
  class Hermitian{
  public:
    const OBJ& obj;
    explicit Hermitian(const OBJ& _obj):obj(_obj){}
  };


  template<typename OBJ>
  class Broadcast{
  public:
    const OBJ& obj;
    explicit Broadcast(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ>
  class Scatter{
  public:
    const OBJ& obj;
    explicit Scatter(const OBJ& _obj):obj(_obj){}
  };

  template<typename OBJ0,typename OBJ1>
  class Outer{
  public:
    const OBJ0& obj0;
    const OBJ1& obj1;
    explicit Outer(const OBJ0& _obj0, const OBJ1& _obj1): 
      obj0(_obj0), obj1(_obj1){}
  };

  template<typename OBJ0,typename OBJ1>
  class Convolve{
  public:
    const OBJ0& obj0;
    const OBJ1& obj1;
    explicit Convolve(const OBJ0& _obj0, const OBJ1& _obj1): 
      obj0(_obj0), obj1(_obj1){}
  };


  // ------------------------------------------------------------------------


  template<typename OBJ>
  Transpose<OBJ> transp(const OBJ& x){
    return Transpose<OBJ>(x);
  }

  template<typename OBJ>
  Conjugate<OBJ> conj(const OBJ& x){
    return Conjugate<OBJ>(x);
  }

  template<typename OBJ>
  Hermitian<OBJ> herm(const OBJ& x){
    return Hermitian<OBJ>(x);
  }


  //template<typename OBJ>
  //Broadcast<OBJ> broadcast(const OBJ& x){
  //return Broadcast<OBJ>(x);
  //}

  template<typename OBJ>
  Scatter<OBJ> scatter(const OBJ& x){
    return Scatter<OBJ>(x);
  }

  //template<typename OBJ0, typename OBJ1>
  //Outer<OBJ0,OBJ1> outer(const OBJ0& x0, const OBJ1& x1){
  //return Outer<OBJ0,OBJ1>(x0,x1);
  //}

  template<typename OBJ0, typename OBJ1>
  Convolve<OBJ0,OBJ1> convolve(const OBJ0& x0, const OBJ1& x1){
    return Convolve<OBJ0,OBJ1>(x0,x1);
  }



  // ---------------------------------------------------------------------------------------------------------
  

  template<typename OP, typename ARR>
  ARR outerprod(const ARR& x, const ARR& y){
    OP op(x,y);
    ARR R(Gdims(x.adims,y.adims),op.spec(),fill::zero);
    R.outer(op.add_prod_op(),x,y);
    return R;
  }

  template<typename OP, typename ARR, typename ARG0>
  ARR outerprod(const ARR& x, const ARR& y, const ARG0& arg0){
    OP op(x,y,arg0);
    ARR R(Gdims(x.adims,y.adims),op.spec(),fill::zero);
    R.outer(op.add_prod_op(),x,y);
    return R;
  }



  template<typename OP, typename ARR>
  ARR convolution(const ARR& x, const ARR& y){
    OP op(x,y);
    ARR R(x.adims.convolve(y.adims),op.spec(),fill::zero);
    R.convolve(op.add_prod_op(),x,y);
    return R;
  }

  template<typename OP, typename ARR, typename ARG0>
  ARR convolution(const ARR& x, const ARR& y, const ARG0& arg0){
    OP op(x,y,arg0);
    ARR R(x.adims.convolve(y.adims),op.spec(),fill::zero);
    R.convolve(op.add_prod_op(),x,y);
    return R;
  }


  template<typename OP, typename ARR>
  ARR matrixprod(const ARR& x, const ARR& y){
    OP op(x,y);
    ARR R(x.adims.Mprod(y.adims),op.spec(),fill::zero);
    R.matrixprod(op.add_prod_op(),x,y);
    return R;
  }

  template<typename OP, typename ARR, typename ARG0>
  ARR matrixprod(const ARR& x, const ARR& y, const ARG0& arg0){
    OP op(x,y,arg0);
    ARR R(x.adims.Mprod(y.adims),op.spec(),fill::zero);
    R.matrixprod(op.add_prod_op(),x,y);
    return R;
  }


}

#endif
