
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3part
#define _GElibSO3part

#include "GElib_base.hpp"
//#include "SO3partA.hpp"
#include "SO3partB.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"
#include "CscalarObj.hpp"
#include "CtensorObj.hpp"


namespace GElib{


  class SO3part_spec{
  public:
    int l,n, nbu, dev;
    SO3part_spec(const int _l, const int _n, const int _nbu=-1, const int _dev=0): 
      l(_l), n(_n), nbu(_nbu), dev(_dev){}
  };


  class SO3part: public GELIB_SO3PART_IMPL{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;
    template<typename TYPE>
    using Transpose=cnine::Transpose<TYPE>;
    template<typename TYPE>
    using Gtensor=cnine::Gtensor<TYPE>;

    typedef cnine::Gdims Gdims;
    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;

    ~SO3part(){
    }


    // ---- Constructors -------------------------------------------------------------------------------------


    using GELIB_SO3PART_IMPL::GELIB_SO3PART_IMPL;


    static SO3part raw(const int _l, const int _n){
      return SO3part(1,_l,_n,cnine::fill::raw,0);}

    static SO3part raw(const int _b, const int _l, const int _n, const int _dev=0){
      return SO3part(_b,_l,_n,cnine::fill::raw,_dev);}

    static SO3part raw_like(const SO3part& x){
      return SO3part(x.getb(), x.getl(), x.getn(), cnine::fill::raw,x.get_dev());}


    static SO3part zero(const int _l, const int _n){
      return SO3part(1,_l,_n,cnine::fill::zero,0);}

    static SO3part zero(const int _b, const int _l, const int _n, const int _dev=0){
      return SO3part(_b,_l,_n,cnine::fill::zero,_dev);}

    static SO3part zero_like(const SO3part& x){
      return SO3part(x.getb(), x.getl(), x.getn(), cnine::fill::zero,x.get_dev());}



    static SO3part gaussian(const int _l, const int _n){
      return SO3part(1,_l,_n,cnine::fill::gaussian,0);}

    static SO3part gaussian(const int _b, const int _l, const int _n, const int _dev=0){
      return SO3part(_b,_l,_n,cnine::fill::gaussian,_dev);}

    static SO3part gaussian_like(const SO3part& x){
      return SO3part(x.getb(), x.getl(), x.getn(), cnine::fill::gaussian,x.get_dev());}


    //static SO3part spharm(const int _l, const float x, const float y, const float z){
    //SO3part R(_l,1,-1,cnine::fill::zero);
    //R.add_spharm(x,y,z);
    //return R;
    //}


    // ---- Lambda constructors -------------------------------------------------------------------------------


    //SO3part(const int _b, const int _l, const int _n, 
    //std::function<complex<float>(const int b, const int i, const int m)> fn):
    //GELIB_SO3PART_IMPL({_b,2*_l+1,_n},[&](const int b, const int i, const int j){return fn(b,j,i-_l);}){}
    

   public: // ---- Copying ------------------------------------------------------------------------------------
    

    SO3part(const SO3part& x):
      GELIB_SO3PART_IMPL(x){}
      
    SO3part(SO3part&& x):
      GELIB_SO3PART_IMPL(std::move(x)){}

    SO3part& operator=(const SO3part& x){
      GELIB_SO3PART_IMPL::operator=(x);
      return *this;
    }

    SO3part& operator=(SO3part&& x){
      GELIB_SO3PART_IMPL::operator=(x);
      return *this;
    }


  public: // ---- Variants ----------------------------------------------------------------------------------

    
    SO3part(const SO3part& x, const int _dev):
      GELIB_SO3PART_IMPL(x,_dev){
    }

    SO3part(const SO3part& x, const device& _dev):
      GELIB_SO3PART_IMPL(x,_dev.id()){
    }

    SO3part(SO3part& x, const cnine::view_flag& flag):
      GELIB_SO3PART_IMPL(x,flag){
    }
      
    
  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3part(const GELIB_SO3PART_IMPL& x):
      GELIB_SO3PART_IMPL(x){
      //cout<<"SO3partA -> SO3part"<<endl;
    }
      
    SO3part(GELIB_SO3PART_IMPL&& x):
      GELIB_SO3PART_IMPL(std::move(x)){
      //cout<<"move SO3partA -> SO3part"<<endl;
    }

    SO3part to(const device& _dev) const{
      return SO3part(*this,_dev);
    }


   public: // ---- Transport ----------------------------------------------------------------------------------
  

    SO3part to_device(const int _dev){
      return SO3part(GELIB_SO3PART_IMPL::to_device(_dev));
    }
  
   
  public: // ---- Access -------------------------------------------------------------------------------------


  public: // ---- Operations ---------------------------------------------------------------------------------


  public: // ---- Accumulating operations ------------------------------------------------------------------



  public: // ---- Spherical harmonics -----------------------------------------------------------------------


    static SO3part  spharm(const int l, const int n, const Gtensor<float>& _x, const int _nbu=-1, const device& dev=0){
      assert(_x.k==1);
      assert(_x.dims[0]==3);
      float x=_x(0);
      float y=_x(1);
      float z=_x(2);
      //return SO3part(engine::new_SO3part_spharm(l,x(0),x(1),x(2),_nbu,dev.id()),l,1,_nbu);

      float length=sqrt(x*x+y*y+z*z); 
      float len2=sqrt(x*x+y*y);
      complex<float> cphi(x/len2,y/len2);

      cnine::Gtensor<float> P=SO3_sphGen(l,z/length);
      vector<complex<float> > phase(l+1);
      phase[0]=complex<float>(1.0,0);
      for(int m=1; m<=l; m++)
	phase[m]=cphi*phase[m-1];
      
      cnine::Gtensor<complex<float> > R({2*l+1,n},cnine::fill::raw);
      for(int m=0; m<=l; m++){
	complex<float> a=phase[m]*complex<float>(P(l,m)); // *(1-2*(m%2))
	complex<float> b=complex<float>(1-2*(m%2))*std::conj(a);
	for(int j=0; j<n; j++){
	  R(l+m,j)=a; 
	  R(l-m,j)=b; 
	}
      }

      return SO3part(R);
      //return newSO3part(engine::new_SO3part(R),l,1);
      //return SO3part(Cengine_engine->push<new_SO3part_from_Gtensor_op>(R,dev.id()),l,n);
    }
    
    /*
    SO3part static spharm(const int l, const int n, const float x, const float y, const float z, const int _nbu=-1, const device_id& dev=0){
      //return SO3part(engine::new_SO3part_spharm(l,x,y,z,_nbu,dev.id()),l,1,_nbu);
      return SO3part(Cengine_engine->push<new_spharm_op>(l,n,x,y,z,_nbu,dev.id()),l,n);
    }
    */
 



  public: // ---- Not in-place operations --------------------------------------------------------------------

    
    //SO3part plus(const SO3part& x){
    //return SO3part(Cengine_engine->push<ctensor_add_op>(hdl,x.hdl,dims),l,n,nbu);
    //}

    //SO3part normalize() const{
      //return SO3part(::Cengine::engine::ctensor_normalize_cols(hdl),l,n,nbu);
    //}

    //void add_normalize_back(const SO3part& g, const SO3part& x){
      //replace(hdl,::Cengine::engine::ctensor_add_normalize_cols_back(hdl,g.hdl,x.hdl));
    //}

    //SO3part chunk(const int i, const int n=1) const {
    //return GELIB_SO3PART_IMPL::chunk(1,i,n);
    //}

    SO3part rotate(const SO3element& r){
      return SO3part(GELIB_SO3PART_IMPL::rotate(r));
    }

    /*
    SO3part rotate(const SO3element& r){
      GELIB_SO3PART_IMPL D(WignerMatrix<float>(getl(),r),dev);
      SO3part R(getl(),getn(),cnine::fill::zero,dev);
      cout<<R<<endl;
      cout<<D<<endl;
      cout<<*this<<endl;
      R.add_prod(D,*this);
      return R;
    }
    */

    /*
    GELIB_SO3PART_IMPL fragment_norms() const{
      return column_norms();
    }

    SO3part divide_fragments(const GELIB_SO3PART_IMPL& N) const{
      return divide_columns(N);
    }
    */


  public: // ---- In-place operators -------------------------------------------------------------------------



  public: // ---- Binary operators ---------------------------------------------------------------------------


    /*
    SO3part operator+(const SO3part& y) const{
      SO3part R(*this);
      R.add(y);
      return R;
    }

    SO3part operator-( const SO3part& y) const{
      SO3part R(*this);
      R.subtract(y);
      return R;
    }

    SO3part operator*(const cscalar& c) const{
      SO3part R(getl(),getn(),nbu,cnine::fill::zero);
      R.add(*this,c);
      return R;
    }

    SO3part operator*(const ctensor& M) const{
      SO3part R(getl(),M.get_dims()[0],cnine::fill::zero);
      R.add_Mprod_AT<0>(*this,M);
      return R;
    }
    
    ctensor operator*(const Transpose<SO3part>& _y) const{
      const SO3part& y=_y.obj;
      assert(y.getl()==getl());
      ctensor R(Gdims(getn(),y.getn()),cnine::fill::zero);
      R.add_mprod_TA(y,*this);
      return R;
    }
    */
    

  public: // ---- I/O --------------------------------------------------------------------------------------


    string classname() const{
      return "GEnet::SO3part";
    }

    string describe() const{
      return "SO3part(l="+to_string(getl())+",n="+to_string(getn())+")";
    }

    string repr(const string indent="") const{
      return indent+"<GElib::SO3part(l="+to_string(getl())+",n="+to_string(getn())+")>";
    }

    friend ostream& operator<<(ostream& stream, const SO3part& x){
      stream<<x.str(); return stream;}

  };


  // ---- Post-class functions -------------------------------------------------------------------------------


  inline SO3part CGproduct(const SO3part& x, const SO3part& y, const int l){
    const int dev=x.get_dev()*y.get_dev();
    SO3part R(x.getb(),l,x.getn()*y.getn(),cnine::fill::zero,dev);
    R.add_CGproduct(x,y,0);
    return R;
  }

  /*  
  inline SO3part DiagCGproduct(const SO3part& x, const SO3part& y, const int l){
    assert(x.getn()==y.getn());
    const int dev=x.get_dev()*y.get_dev();
    SO3part R(l,x.getn(),cnine::fill::zero,dev);
    R.add_DiagCGproduct(x,y,0);
    return R;
  }
  */


}

#endif 


