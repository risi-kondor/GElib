
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibBatchedSO3part
#define _GElibBatchedSO3part

#include "GElib_base.hpp"
//#include "TensorView.hpp"
#include "TensorVirtual.hpp"
#include "BatchedSO3partView.hpp"
#include "SO3templates.hpp"


namespace GElib{

  template<typename TYPE>
  class BatchedSO3part: public cnine::TensorVirtual<complex<TYPE>, BatchedSO3partView<TYPE> >{
  public:

    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;

    typedef cnine::TensorVirtual<complex<TYPE>, BatchedSO3partView<TYPE> > VTensor;
    typedef BatchedSO3partView<TYPE> BatchedSO3partView;

    using VTensor::VTensor;
    using VTensor::dims;
    using VTensor::operator*;

    using BatchedSO3partView::getl;
    using BatchedSO3partView::getn;
    using BatchedSO3partView::dim;


  public: // ---- Constructors -------------------------------------------------------------------------------


    BatchedSO3part(const int b, const int l, const int n, const int _dev=0):
      BatchedSO3part(b,Gdims({2*l+1,n}),_dev){}


  public: // ---- Named constructors -------------------------------------------------------------------------

    
    static BatchedSO3part zero(const int b, const int l, const int n, const int _dev=0){
      return BatchedSO3part(b,Gdims({2*l+1,n}),cnine::fill_zero(),_dev);}
    
    static BatchedSO3part sequential(const int b, const int l, const int n, const int _dev=0){
      return BatchedSO3part(b,Gdims({2*l+1,n}),cnine::fill_sequential(),_dev);}
    
    static BatchedSO3part gaussian(const int b, const int l, const int n, const int _dev=0){
      return BatchedSO3part(b,Gdims({2*l+1,n}),cnine::fill_gaussian(),_dev);}
    

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
      return "GElib::BatchedSO3part";
    }

    //string repr(const string indent="") const{
    //return "<GElib::BatchedSO3part(l="+to_string(getl())+",n="+to_string(getn())+")>";
    //}
    
    friend ostream& operator<<(ostream& stream, const BatchedSO3part& x){
      stream<<x.str(); return stream;
    }


  };


  template<typename TYPE>
  inline BatchedSO3part<TYPE> CGproduct(const BatchedSO3partView<TYPE>& x, const BatchedSO3partView<TYPE>& y, const int l){
      assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
      BatchedSO3part<TYPE> R=BatchedSO3part<TYPE>::zero(x.getb(),l,x.getn()*y.getn(),x.device());
      add_CGproduct(R,x,y);
      return R;
    }

  template<typename TYPE>
  inline BatchedSO3part<TYPE> DiagCGproduct(const BatchedSO3partView<TYPE>& x, const BatchedSO3partView<TYPE>& y, const int l){
      assert(x.getn()==y.getn());
      assert(l>=abs(x.getl()-y.getl()) && l<=x.getl()+y.getl());
      BatchedSO3part<TYPE> R=BatchedSO3part<TYPE>::zero(x.getb(),l,x.getn(),x.device());
      add_DiagCGproduct(R,x,y);
      return R;
    }




}


#endif 
