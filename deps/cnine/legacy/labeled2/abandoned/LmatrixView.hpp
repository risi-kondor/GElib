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


#ifndef _CnineLmatrixView
#define _CnineLmatrixView

#include "Cnine_base.hpp"
#include "TensorView.hpp"
#include "Llist.hpp"


namespace cnine{

  template<typename LABELS0, typename LABELS1, typename TYPE>
  class LmatrixView: public TensorView<TYPE>{
  public:

    typedef TensorView<TYPE> BASE;

    Llist<LABELS0> labels0;
    Llist<LABELS1> labels1;



  public: // ---- Constructors ------------------------------------------------------------------------------


    LmatrixView(){}

    LmatrixView(const Llist<LABELS0>& _labels0, const Llist<LABELS1>& _labels1, const int _dev=0): 
      TensorView<TYPE>(Gdims({_labels0.isize(),_labels1.isize()}),_dev),
      labels0(_labels0),
      labels1(_labels1){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    LmatrixView(const Llist<LABELS0>& _labels0, const Llist<LABELS1>& _labels1, const FILLTYPE& fill, const int _dev=0): 
      TensorView<TYPE>(Gdims({_labels0.isize(),_labels1.isize()}),fill,_dev),
      labels0(_labels0),
      labels1(_labels1){}


  public: // ---- Access ------------------------------------------------------------------------------------


    TYPE operator()(const LABELS0& i0, const LABELS1& i1){
      return BASE::operator()(labels0(i0),labels1(i1));
    }
    
    void set(const LABELS0& i0, const LABELS1& i1, const TYPE v){
      BASE::set(labels0(i0),labels1(i1),v);
    }

    void inc(const LABELS0& i0, const LABELS1& i1, const TYPE v){
      BASE::inc(labels0(i0),labels1(i1),v);
    }
    

  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Rows: "<<labels0<<endl;
      oss<<indent<<"Cols: "<<labels1<<endl;
      oss<<TensorView<TYPE>::str(indent);
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LmatrixView<LABELS0,LABELS1,TYPE>& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
