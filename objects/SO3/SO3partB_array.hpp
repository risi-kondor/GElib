// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2022, Imre Risi Kondor 
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _SO3partB_array
#define _SO3partB_array

#include "CtensorB_array.hpp"
#include "SO3part3_view.hpp"
#include "SO3part_addCGproductFn.hpp"
#include "SO3part_addCGproduct_back0Fn.hpp"
#include "SO3part_addCGproduct_back1Fn.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"

extern GElib::SO3_CGbank SO3_cgbank;
extern GElib::SO3_SPHgen SO3_sphGen;



namespace GElib{


  // An SO3partB_array is an (a1 x ... x ak) x N x b x (2l+1) x n   dimensional complex tensor.


  class SO3partB_array: public cnine::CtensorB_array{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::Gdims Gdims;
    typedef cnine::Gstrides Gstrides;
    typedef cnine::Ctensor3_view Ctensor3_view;

    using CtensorB_array::CtensorB_array;

    
  public: // ---- Constructors -------------------------------------------------------------------------------


    //SO3partB_array(const int N, const int b, const int l, const int n, const int _dev=0):
    //CtensorB_array({N},{b,2*l+1,n},_dev){}

    //template<typename FILLTYPE, typename = typename 
    //     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //SO3partB_array(const int N, const int b, const int l, const int n, const FILLTYPE& dummy, const int _dev=0):
    //CtensorB_array({N},{b,2*l+1,n},dummy,_dev){}

    SO3partB_array(const Gdims& _adims, const int l, const int n, const int _dev=0):
      CtensorB_array(_adims,{2*l+1,n},_dev){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partB_array(const Gdims& _adims, const int l, const int n, const FILLTYPE& dummy, const int _dev=0):
      CtensorB_array(_adims,{2*l+1,n},dummy,_dev){}

    

  public: // ---- Named constructors -------------------------------------------------------------------------

    
    //static SO3partB_array zero(const int N, const int b, const int l, const int n,  const int _dev=0){
    //return SO3partB_array(N,b,l,n,cnine::fill_zero(),_dev);
    //}

    //static SO3partB_array gaussian(const int N, const int b, const int l, const int n,  const int _dev=0){
    //return SO3partB_array(N,b,l,n,cnine::fill_gaussian(),_dev);
    //}


    static SO3partB_array zero(const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partB_array(_adims,l,n,cnine::fill_zero(),_dev);
    }

    static SO3partB_array ones(const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partB_array(_adims,l,n,cnine::fill_zero(),_dev);
    }

    static SO3partB_array gaussian(const Gdims& _adims, const int l, const int n,  const int _dev=0){
      return SO3partB_array(_adims,l,n,cnine::fill_gaussian(),_dev);
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partB_array(const CtensorB& x):
      CtensorB_array(x,-2){}
      
    SO3partB_array(CtensorB&& x):
      CtensorB_array(std::move(x),-2){}
      

  public: // ---- Access -------------------------------------------------------------------------------------


    int getN() const{
      return memsize/strides.back(2);
    }
    
    //int getb() const{
    //return dims.back(2);
    //}

    int getl() const{
      return (dims.back(1)-1)/2;
    }

    int getn() const{
      return dims.back(0);
    }

    //Gdims get_cdims(){
    //return Gdims({getb(),getl(),getn()});
    //}

    //Gstrides get_cstrides(){
    //return strides.chunk(strides.size()-3);
    //}

        
  public: // ---- Access views --------------------------------------------------------------------------------

    /*
    Ctensor3_view view3D() const{
      return Ctensor3_view(arr,Gdims({dims(0)*dims(1),dims(2),dims(3)}),Gstrides({strides[1],strides[2],strides[3]}),coffs);
    }
    */

    /*
    SO3part3_view Pview3D() const{
      if(dev==0) return SO3part3_view(arr,Gdims({GetN(),getl(),getn()}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs);
      else return SO3part3_view(arrg,Gdims({getN(),getl(),getn()}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs,dev);
    }
    */

    SO3part3_view part3_view() const{
      if(dev==0) return SO3part3_view(arr,Gdims({getN(),dims.back(1),dims.back(0)}),
	Gstrides({strides.back(2),strides.back(1),strides.back(0)}),coffs);
      else return SO3part3_view(arrg,Gdims({getN(),dims.back(1),dims.back(0)}),
	Gstrides({strides.back(2),strides.back(1),strides.back(0)}),coffs,dev);
    }

    /*
    SO3part3_view cell_view(const int i) const{
      return SO3part3_view(arr+strides[0]*i,{dims(1),dims(2),dims(3)},{strides[1],strides[2],strides[3]},coffs,dev);
    }
    */

    //operator SO3part3_view() const{
    //if(dev==0) return SO3part3_view(arr,get_cdims(),get_cstrides(),coffs);
    //else return SO3part3_view(arrg,get_cdims(),get_cstrides(),coffs,dev);
    //}

    /*
    SO3Fpart3_view Fview3D() const{
      if(dev==0) return SO3Fpart3_view(arr,Gdims({dims(0)*dims(1),dims(2),dims(3)}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs);
      else return SO3Fpart3_view(arrg,Gdims({dims(0)*dims(1),dims(2),dims(3)}),
	Gstrides({strides[1],strides[2],strides[3]}),coffs,dev);
    }
    */

  public: // ---- Operations ---------------------------------------------------------------------------------


    //SO3partB_array operator-(const SO3partB_array& y) const{
    //return (*this)-y;
    //}


  public: // ---- Rotations ----------------------------------------------------------------------------------


    SO3partB_array rotate(const SO3element& r){
      CtensorB D(WignerMatrix<float>(getl(),r),dev);
      SO3partB_array R=SO3partB_array::zero(get_adims(),getl(),getn(),dev);
      cout<<R.repr()<<endl;

      auto dv=D.view2D();
      //auto xv=this->part3_view();
      //auto rv=R.part3_view();
      auto xv=this->view3();
      auto rv=R.view3();
      
      int B=rv.n0;
      for(int b=0; b<B; b++){
	auto v=rv.slice0(b);
	cnine::Ctensor_add_mprod_AA()(v,dv,xv.slice0(b));
      }

      return R;
    }



  public: // ---- Cumulative Operations ----------------------------------------------------------------------


    void add_gather(const SO3partB_array& x, const cnine::Rmask1& mask){
      CtensorB::add_gather(x,mask);
    }


  public: // ---- CG-products --------------------------------------------------------------------------------

    
    SO3partB_array CGproduct(const SO3partB_array& y, const int l) const{
      assert(l>=abs(getl()-y.getl()) && l<=getl()+y.getl());
      SO3partB_array R=SO3partB_array::zero(get_adims(),l,getn()*y.getn(),get_dev());
      R.add_CGproduct(*this,y);
      return R;
    }


    void add_CGproduct(const SO3partB_array& x, const SO3partB_array& y, const int _offs=0){
      auto v=this->part3_view();
      cout<<v.n0<<v.n1<<v.n2<<endl;
      SO3part_addCGproductFn()(v,x.part3_view(),y.part3_view(),_offs);
    }

    void add_CGproduct_back0(const SO3partB_array& g, const SO3partB_array& y, const int _offs=0){
      auto v=this->part3_view();
      SO3part_addCGproduct_back0Fn()(v,g.part3_view(),y.part3_view(),_offs);
    }

    void add_CGproduct_back1(const SO3partB_array& g, const SO3partB_array& x, const int _offs=0){
      auto v=this->part3_view();
      SO3part_addCGproduct_back1Fn()(v,g.part3_view(),x.part3_view(),_offs);
    }


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
    
  };

}

#endif 
