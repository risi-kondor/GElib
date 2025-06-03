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


#ifndef _CnineLtensorViewSub
#define _CnineLtensorViewSub

#include "Cnine_base.hpp"
#include "LtensorView.hpp"


namespace cnine{

  template<typename TYPE, typename CLASS>
  class LtensorViewSub: public LtensorView<TYPE>{
  public:

    typedef LtensorView<TYPE> BASE;

    using BASE::BASE;
    using BASE::arr;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;
    using BASE::labels;

    using BASE::is_batched;
    using BASE::nbatch;

    using BASE::is_gridded;
    using BASE::gdims;


  public: // ---- Constructors ------------------------------------------------------------------------------


    // inherited 
    //LtensorViewSub(const Gdims& _dims, const DimLabels& _labels, const int fcode, const int _dev=0):
    //BASE(_dims,fcode,_dev), 
    //labels(_labels){}

    // inherited
    //LtensorViewSub(const MemArr<TYPE>& _arr, const Gdims& _dims, const GstridesB& _strides, const DimLabels& _labels):
    //BASE(_arr,_dims,_strides),
    //labels(_labels){}
    


  public: // ---- Access ------------------------------------------------------------------------------------




  public: // ---- Batches -----------------------------------------------------------------------------------


    bool is_batched() const{
      return labels._batched();
    }

    int nbatch() const{
      CNINE_ASSRT(is_batched());
      return dims[0];
    }

    LtensorViewSub batch(const int i) const{
      return CLASS(BASE::batch(i));
    }

    void for_each_batch(const std::function<void(const int, const CLASS& x)>& lambda) const{
      int B=nbatch();
      for(int b=0; b<B; b++)
	lambda(b,batch(b));
    }


  public: // ---- Grid ---------------------------------------------------------------------------------------


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string classname() const{
      return "LtensorViewSub";
    }

    /*
    string describe() const{
      ostringstream oss;
      oss<<"LtensorViewSub"<<dims<<" ["<<strides<<"]"<<endl;
      return oss.str();
    }

    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Ltensor"<<labels.str(dims)<<"["<<dev<<"]:"<<endl;
      oss<<BASE::str(indent);
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const LtensorViewSub<TYPE>& x){
      stream<<x.str(); return stream;
    }
    */

  };


}

#endif 
