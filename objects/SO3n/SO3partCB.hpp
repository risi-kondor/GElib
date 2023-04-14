
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partCB
#define _GElibSO3partCB

#include "GElib_base.hpp"
//#include "TensorView.hpp"
#include "TensorVirtual.hpp"
#include "SO3partViewB.hpp"


namespace GElib{

  template<typename TYPE>
  class SO3partCB: public cnine::TensorVirtual<complex<TYPE>, SO3partViewB<TYPE> >{
  public:

    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;

    typedef cnine::TensorVirtual<complex<TYPE>, SO3partViewB<TYPE> > VTensor;
    typedef SO3partViewB<TYPE> SO3partViewB;

    using VTensor::VTensor;
    using VTensor::dims;
    using VTensor::operator*;

    using SO3partViewB::getl;
    using SO3partViewB::getn;
    using SO3partViewB::dim;


  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3partCB(const int b, const int l, const int n, const int _dev=0):
      SO3partCB(b,Gdims({2*l+1,n}),_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static SO3partCB zero(const int b, const int l, const int n, const int _dev=0){
      return SO3partCB(b,Gdims({2*l+1,n}),cnine::fill_zero(),_dev);}
    
    static SO3partCB sequential(const int b, const int l, const int n, const int _dev=0){
      return SO3partCB(b,Gdims({2*l+1,n}),cnine::fill_sequential(),_dev);}
    
    static SO3partCB gaussian(const int b, const int l, const int n, const int _dev=0){
      return SO3partCB(b,Gdims({2*l+1,n}),cnine::fill_gaussian(),_dev);}
    

  public: // ---- Access -------------------------------------------------------------------------------------


    //int getl() const{
    //return (dims(0)-1)/2;
    //}

    //int getn() const{
    //return dims(1);
    //}

    bool is_F() const{
      return (dim(0)==dim(1));
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    
  public: // ---- CG-products --------------------------------------------------------------------------------

    


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GElib::SO3partCB";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3partCB(l="+to_string(getl())+",n="+to_string(getn())+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3partCB& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename TYPE>
  inline SO3partCB<TYPE> CGproduct(const SO3partViewB<TYPE>& x, const SO3partViewB<TYPE>& y, const int l){
      assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
      SO3partCB<TYPE> R=SO3partCB<TYPE>::zero(x.getb(),l,x.getn()*y.getn(),x.device());
      R.add_CGproduct(x,y);
      return R;
    }




}


#endif 
