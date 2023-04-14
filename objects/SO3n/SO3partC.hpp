
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partC
#define _GElibSO3partC

#include "GElib_base.hpp"
//#include "TensorView.hpp"
#include "TensorVirtual.hpp"
#include "SO3partView.hpp"


namespace GElib{

  template<typename TYPE>
  class SO3partC: public cnine::TensorVirtual<complex<TYPE>, SO3partView<TYPE> >{
  public:

    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;

    typedef cnine::TensorVirtual<complex<TYPE>, SO3partView<TYPE> > TensorVirtual;
    using TensorVirtual::TensorVirtual;
    using TensorVirtual::dims;


    using TensorVirtual::operator*;

    //using TensorVirtual::getl;
    using SO3partView<TYPE>::getl;
    using SO3partView<TYPE>::getn;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3partC(const int l, const int n, const int _dev=0):
      SO3partC(Gdims({2*l+1,n}),_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3partC<TYPE> zero(const int l, const int n, const int _dev=0){
      return SO3partC<TYPE>(Gdims({2*l+1,n}),cnine::fill_zero(),_dev);}
    
    static SO3partC<TYPE> sequential(const int l, const int n, const int _dev=0){
      return SO3partC<TYPE>(Gdims({2*l+1,n}),cnine::fill_sequential(),_dev);}
    
    static SO3partC<TYPE> gaussian(const int l, const int n, const int _dev=0){
      return SO3partC<TYPE>(Gdims({2*l+1,n}),cnine::fill_gaussian(),_dev);}
    

  public: // ---- Access -------------------------------------------------------------------------------------


    //int getl() const{
    //return (dims(0)-1)/2;
    //}

    //int getn() const{
    //return dims(1);
    //}

    bool is_F() const{
      return (dims(0)==dims(1));
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    
  public: // ---- CG-products --------------------------------------------------------------------------------

    


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3partC";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3partC(l="+to_string(getl())+",n="+to_string(getn())+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3partC& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename TYPE>
  inline SO3partC<TYPE> CGproduct(const SO3partView<TYPE>& x, const SO3partView<TYPE>& y, const int l){
      assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
      SO3partC<TYPE> R=SO3partC<TYPE>::zero(l,x.getn()*y.getn(),x.device());
      R.add_CGproduct(x,y);
      return R;
    }




}


#endif 
