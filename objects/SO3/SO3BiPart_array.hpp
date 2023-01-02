// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2023, Imre Risi Kondor 
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3BiPart_array
#define _SO3BiPart_array

#include "CtensorArrayB.hpp"
#include "SO3part3_view.hpp"
#include "SO3partB.hpp"

#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"
#include "SO3part_addCGsquareFn.hpp"
#include "SO3part_addFproduct_Fn.hpp"
#include "SO3part_addFproduct_back0Fn.hpp"
#include "SO3part_addFproduct_back1Fn.hpp"

#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{


  // An SO3BiPart_array is an (a1 x ... x ak) x (2*l1+1) x (2*l2+1) x n   dimensional complex tensor.


  class SO3BiPart_array: public cnine::CtensorArrayB{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;
    typedef cnine::Gstrides Gstrides;
    typedef cnine::Ctensor3_view Ctensor3_view;

    using CtensorArrayB::CtensorArrayB;

    
  public: // ---- Constructors -------------------------------------------------------------------------------


    SO3BiPart_array(const Gdims& _adims, const int l1, const int l2, const int n, const int _dev=0):
      CtensorArrayB(_adims,{2*l1+1,2*l2+1,n},_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3BiPart_array(const Gdims& _adims, const int l1, const int l2, const int n, const FILLTYPE& dummy, const int _dev=0):
      CtensorArrayB(_adims,{2*l1+1,2*l2+1,n},dummy,_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3BiPart_array(const int b, const Gdims& _adims, const int l1, const int l2, const int n, const FILLTYPE& dummy, const int _dev=0):
      CtensorArrayB(_adims.prepend(b),{2*l1+1,2*l2+1,n},dummy,_dev){
      batched=true;
    }

    

  public: // ---- Named constructors -------------------------------------------------------------------------


    static SO3BiPart_array zero(const Gdims& _adims, const int l1, const int l2, const int n,  const int _dev=0){
      return SO3BiPart_array(_adims,l1,l2,n,cnine::fill_zero(),_dev);
    }

    static SO3BiPart_array ones(const Gdims& _adims, const int l1, const int l2, const int n,  const int _dev=0){
      return SO3BiPart_array(_adims,l1,l2,n,cnine::fill_zero(),_dev);
    }

    static SO3BiPart_array gaussian(const Gdims& _adims, const int l1, const int l2, const int n,  const int _dev=0){
      return SO3BiPart_array(_adims,l1,l2,n,cnine::fill_gaussian(),_dev);
    }


    static SO3BiPart_array zero(const int b, const Gdims& _adims, const int l1, const int l2, const int n,  const int _dev=0){
      return SO3BiPart_array(b,_adims,l1,l2,n,cnine::fill_zero(),_dev);
    }

    static SO3BiPart_array ones(const int b, const Gdims& _adims, const int l1, const int l2, const int n,  const int _dev=0){
      return SO3BiPart_array(b,_adims,l1,l2,n,cnine::fill_zero(),_dev);
    }

    static SO3BiPart_array gaussian(const int b, const Gdims& _adims, const int l1, const int l2, const int n,  const int _dev=0){
      return SO3BiPart_array(b,_adims,l1,l2,n,cnine::fill_gaussian(),_dev);
    }

    
    static SO3BiPart_array zeros_like(const SO3BiPart_array& x){
      return SO3BiPart_array(x.get_adims(),x.getl1(),x.getl2(),x.getn(),cnine::fill_zero(),x.dev);
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    SO3BiPart_array(const SO3BiPart_array& x):
      CtensorArrayB(x){
      GELIB_COPY_WARNING();
    }
      
    SO3BiPart_array(SO3BiPart_array&& x):
      CtensorArrayB(std::move(x)){
      GELIB_MOVE_WARNING();
    }

      
  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3BiPart_array(const CtensorB& x):
      CtensorArrayB(x,-3){
      GELIB_CONVERT_WARNING(x);
    }
      
    SO3BiPart_array(CtensorB&& x):
      CtensorArrayB(std::move(x),-3){
      GELIB_MCONVERT_WARNING(x);
    }

    SO3BiPart_array(const CtensorArrayB& x):
      CtensorArrayB(x){
      GELIB_CONVERT_WARNING(x);
    }
      
    SO3BiPart_array(CtensorArrayB&& x):
      CtensorArrayB(std::move(x)){
      GELIB_MCONVERT_WARNING(x);
    }

      
  public: // ---- ATen --------------------------------------------------------------------------------------



  public: // ---- Access -------------------------------------------------------------------------------------


    int getN() const{
      return memsize/strides.back(3);
    }
    
    //int getb() const{
    //return dims(0);
    //}

    int getl1() const{
      return (dims.back(2)-1)/2;
    }

    int getl2() const{
      return (dims.back(2)-1)/2;
    }

    int getn() const{
      return dims.back(0);
    }

    Gdims get_adims() const{
      return dims.chunk(0,ak);
    }

        
  public: // ---- Access views --------------------------------------------------------------------------------


    SO3partB fused_view(){
      return SO3partB(view_fusing_first(ak));
    }

    const SO3partB fused_view() const{
      return SO3partB(const_cast<SO3BiPart_array*>(this)->view_fusing_first(ak));
    }

    //SO3part3_view part4_view() const{
    //SO3BiPart4_view(true_arr(),Gdims({getN(),dims.back(2),dims.back(1),dims.back(0)}),
    //Gstrides({strides.back(3),strides.back(2),strides.back(1),strides.back(0)}),coffs,dev);
    //}


  public: // ---- Operations ---------------------------------------------------------------------------------



  public: // ---- Rotations ----------------------------------------------------------------------------------


    /*
    SO3BiPart_array rotate(const SO3element& r) const{
      CtensorB D(WignerMatrix<float>(getl(),r),dev);
      SO3BiPart_array R=SO3BiPart_array::zeros_like(*this);

      auto dv=D.view2D();
      auto xv=fused_view().view3();
      auto rv=R.fused_view().view3();
      
      int B=rv.n0;
      for(int b=0; b<B; b++){
	auto v=rv.slice0(b);
	cnine::Ctensor_add_mprod_AA()(v,dv,xv.slice0(b));
      }

      return R;
    }
    */


  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add_gather(const SO3BiPart_array& x, const cnine::Rmask1& mask){
      CtensorArrayB::add_gather(x,mask);
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    


  public: // ---- I/O ----------------------------------------------------------------------------------------


    /*
    string str(const string indent="") const{
      ostringstream oss;
      int N=getN();
      for(int i=0; i<N; i++){
	oss<<indent<<"Cell "<<i<<":"<<endl;
	oss<<cell_view(i).str(indent+"  ")<<endl;
      }
      return oss.str();
    }
    */
    
    string classname() const{
      return "GElib::SO3BiPart_array";
    }

    string repr(const string indent="") const{
      return "<GElib::SO3BiPart_array of type ("+get_adims().str()+","+to_string(getl1())+","+
	to_string(getl2())+","+to_string(getn())+")>";
    }
    
    friend ostream& operator<<(ostream& stream, const SO3BiPart_array& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
