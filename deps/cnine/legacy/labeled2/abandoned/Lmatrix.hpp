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


#ifndef _CnineLmatrix
#define _CnineLmatrix

#include "Cnine_base.hpp"
#include "LmatrixView.hpp"


namespace cnine{

  template<typename LABELS0, typename LABELS1, typename TYPE>
  class Lmatrix: public LmatrixView<LABELS0,LABELS1,TYPE>{
  public:

    typedef LmatrixView<LABELS0,LABELS1,TYPE> BASE;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;

    //using BASE::operator=;
    using BASE::ndims;
    using BASE::dim;
    using BASE::set;
    using BASE::transp;

    using BASE::labels0;
    using BASE::labels1;


  public: // ---- Constructors ------------------------------------------------------------------------------


  public: // ---- Constructors ------------------------------------------------------------------------------


    //Tensor():
    //TensorView<TYPE>(MemArr<TYPE>(1),{1},{1}){}

    Lmatrix(){}

    Lmatrix(const Llist<LABELS0>& _labels0, const Llist<LABELS1>& _labels1, const int _dev=0): 
      BASE(_labels0,_labels1,_dev){}


  public: // ---- Named constructors ------------------------------------------------------------------------


    static Lmatrix zero(const Llist<LABELS0>& _labels0, const Llist<LABELS1>& _labels1, const int _dev=0){
      return Lmatrix(_labels0,_labels1,fill_zero(),_dev);
    }

    static Lmatrix constant(const Llist<LABELS0>& _labels0, const Llist<LABELS1>& _labels1, const TYPE v, const int _dev=0){
      return Lmatrix(_labels0,_labels1,fill_constant(v),_dev);
    }

    static Lmatrix identity(const Llist<LABELS0>& _labels0, const Llist<LABELS1>& _labels1, const int _dev=0){
      return Lmatrix(_labels0,_labels1,fill_identity(),_dev);
    }

    static Lmatrix sequential(const Llist<LABELS0>& _labels0, const Llist<LABELS1>& _labels1, const int _dev=0){
      return Lmatrix(_labels0,_labels1,fill_sequential(),_dev);
    }

    static Lmatrix gaussian(const Llist<LABELS0>& _labels0, const Llist<LABELS1>& _labels1, const int _dev=0){
      return Lmatrix(_labels0,_labels1,fill_gaussian(),_dev);
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "Ltensor";
    }

    string describe() const{
      ostringstream oss;
      oss<<"Ltensor"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Lmatrix<LABELS0,LABELS1,TYPE>& x){
      stream<<x.str(); return stream;
    }


  };
}

#endif 


    /*
    Lmatrix(const Llist<LABELS0>& _labels0, const Llist<LABELS1>& _labels1, const int _dev=0): 
      TensorView<TYPE>(MemArr<TYPE>(_labels0.size()*_labels1.size(),_dev),
	Gdims({_labels0.size(),_labels1.size()}),
	GstridesB(Gdims({_labels0.size(),_labels1.size()}))),
      labels0(_labels0),
      labels1(_labels1){}
    */
