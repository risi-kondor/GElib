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


#ifndef _CnineLtensor
#define _CnineLtensor

#include "Cnine_base.hpp"
#include "LtensorView.hpp"
//#include "LtensorGen.hpp"
#include "TensorSpec.hpp"


namespace cnine{


  template<typename TYPE>
  class Ltensor;

  template<typename TYPE>
  class TensorSpec: public TensorSpecBase<TensorSpec<TYPE> >{
  public:

    typedef TensorSpecBase<TensorSpec<TYPE> > BASE;
    TensorSpec(){}
    TensorSpec(const BASE& x): BASE(x){}

    Ltensor<TYPE> operator()(){
      return Ltensor<TYPE>(*this);
    }
    
  };



  template<typename TYPE>
  class Ltensor: public LtensorView<TYPE>{
  public:

    typedef LtensorView<TYPE> BASE;

    using BASE::BASE;
    using BASE::dims;
    using BASE::strides;
    using BASE::dev;
    using BASE::ndims;
    using BASE::dim;
    using BASE::set;
    using BASE::transp;


  public: // ---- Constructors ------------------------------------------------------------------------------


    Ltensor(): 
      Ltensor({1},DimLabels(),0,0){}

    Ltensor(const TensorSpec<TYPE>& g):
      Ltensor(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()){}

    static TensorSpec<TYPE> make() {return TensorSpec<TYPE>();}
    static TensorSpec<TYPE> raw() {return TensorSpec<TYPE>().raw();}
    static TensorSpec<TYPE> zero() {return TensorSpec<TYPE>().zero();}
    static TensorSpec<TYPE> sequential() {return TensorSpec<TYPE>().sequential();}
    static TensorSpec<TYPE> gaussian() {return TensorSpec<TYPE>().gaussian();}


  public: // ---- Copying -----------------------------------------------------------------------------------


    /*
    Ltensor(const Ltensor<TYPE>& x):
      BASE(x.dims,x.strides), labels(x.labels){
      CNINE_COPY_WARNING();
      view()=x.view();
    }
        
    Ltensor(const Ltensor<TYPE>& x, const nowarn_flag& dummy):
      BASE(x.dims,x.dev){
      view()=x.view();
    }
        
    Ltensor(Ltensor<TYPE>&& x):
      TensorView<TYPE>(x.arr,x.dims,x.strides){
      CNINE_MOVE_WARNING();
      }
    */

  };

}

#endif 


    //template<typename FILLTYPE, typename = typename 
    //std::enable_if<std::is_base_of<cnine::fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //Ltensor(const Gdims& _dims, const DimLabels& _labels, const FILLTYPE& fill, const int _dev=0):
    //BASE(_dims,_labels,fill,_dev){}

    //Ltensor(const Gdims& _dims, const DimLabels& _labels, const int fcode, const int _dev){
    //switch(fcode){
    //default: 
    //}
  //typedef TensorMaker<LtensorView<TYPE> > TensorSpec<TYPE>
  /*
  template<typename CLASS>
  class TensorMaker: public TensorSpecBase<TensorMaker<CLASS> >{
  public:

    TensorMaker(){}

    TensorMaker(const TensorSpecBase<TensorMaker<CLASS> >& x):
      TensorSpecBase<TensorMaker<CLASS> >(x){}

    explicit operator CLASS(){
      return CLASS(*this);
    }

    TensorMaker glow(){
      cout<<"glow"<<endl;
      return *this;
    }

  };


  template<typename TYPE>
  class Ltensor;

  template<typename TYPE>
  class TensorSpec: public TensorMaker<Ltensor<TYPE> >{
  public: 
    TensorSpec(){}
    explicit TensorSpec(const TensorMaker<Ltensor<TYPE> >& x): 
      TensorMaker<Ltensor<TYPE> >(x){}
  };
  */
  /*
  class TensorSpec: public TensorSpecBase<TensorSpec>{
  public:

    TensorSpec(){}

    TensorSpec(const TensorSpecBase<TensorSpec>& x):
      TensorSpecBase<TensorSpec>(x){}

    TensorSpec glow(){
      cout<<"glow"<<endl;
      return *this;
    }

  };
  */
  /*
  template<typename SPEC>
  class TensorSpecifier: public TensorSpecBase<SPEC>{
  public:

    typedef TensorSpecBase<SPEC> BASE;
    //TensorSpecifier(){}
    //TensorSpecifier(const TensorSpecifier& x)=delete;
    //TensorSpecifier& operator=(const TensorSpecifier& x)=delete;

    SPEC glow(){
      cout<<"glow"<<endl;
      return *this;
    }

  };


  template<typename CLASS>
  class TensorMaker: public TensorSpecifier<TensorMaker<CLASS> >{
  public:


    typedef TensorSpecifier<TensorMaker<CLASS> > BASE;
    //using BASE::BASE;
    TensorMaker(){}
    //TensorMaker(const TensorMaker& x)=delete;
    //TensorMaker& operator=(const TensorMaker& x)=delete;

    TensorMaker(const BASE& x):
      BASE(x){}

    explicit operator CLASS(){
      return CLASS(*this);
    }

  };
  */
  /*
  class TensorSpec: public TensorSpecifier<TensorSpec>{
  public:

    typedef TensorSpecifier<TensorSpec> BASE;
    using BASE::BASE;
    TensorSpec(){}

  };
  */

    //template<typename DUMMY>
    //Ltensor(const TensorSpecifier<DUMMY>& g):
    //Ltensor(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()){}

    //Ltensor(const TensorMaker<Ltensor<TYPE> >& g):
    //Ltensor(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()){}

    //Ltensor(const TensorSpec& g):
    //Ltensor(g.get_dims(), g.get_labels(), g.get_fcode(), g.get_dev()){}


    //static TensorMaker<Ltensor<TYPE> > make(){
    //return TensorMaker<Ltensor<TYPE> >();
    //}
  /*
  class TensorMaker: public TensorSpecBase<TensorMaker>{
  public:

    TensorSpec(){}

    TensorSpec(const TensorSpecBase<TensorSpec>& x):
      TensorSpecBase<TensorSpec>(x){}

    //operator Ltensor<float>(){
    //return Ltensor<float>(*this);
    //}
			    
    TensorSpec glow(){
      cout<<"glow"<<endl;
      return *this;
    }

  };
  */

