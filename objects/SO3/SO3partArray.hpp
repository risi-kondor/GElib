
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3partArray
#define _GElibSO3partArray

#include "GElib_base.hpp"
#include "SO3partArrayA.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"
#include "CscalarObj.hpp"
#include "CtensorObj.hpp"
#include "CtensorArray.hpp"
#include "OperationTemplates.hpp"


namespace GElib{


  class SO3partArray: public GELIB_SO3PARTARRAY_IMPL{
  public:

    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    template<typename TYPE>
    using Transpose=cnine::Transpose<TYPE>;
    template<typename TYPE>
    using Scatter=cnine::Scatter<TYPE>;

    template<typename TYPE>
    using Gtensor=cnine::Gtensor<TYPE>;

    typedef cnine::Gdims Gdims;
    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorArray ctensor_arr;
    typedef cnine::RscalarObj Rscalar;
    typedef cnine::CscalarObj Cscalar;
    typedef cnine::CtensorObj Ctensor;
    typedef cnine::CtensorArray CtensorArray;


    ~SO3partArray(){
    }


    // ---- Constructors -------------------------------------------------------------------------------------


    using GELIB_SO3PARTARRAY_IMPL::GELIB_SO3PARTARRAY_IMPL;


    static SO3partArray zero(const Gdims& _adims, const int _l, const int _n, const int _nbu=-1){
      return SO3partArray(_adims,_l,_n,_nbu,cnine::fill::zero);}
    static SO3partArray zero(const Gdims& _adims, const int _l, const int _n, const int _nbu, const device& _dev){
      return SO3partArray(_adims,_l,_n,_nbu,cnine::fill::zero,_dev.id());}

    static SO3partArray ones(const Gdims& _adims, const int _l, const int _n, const int _nbu=-1){
      return SO3partArray(_adims,_l,_n,_nbu,cnine::fill::ones);}
    static SO3partArray ones(const Gdims& _adims, const int _l, const int _n, const int _nbu, const device& _dev){
      return SO3partArray(_adims,_l,_n,_nbu,cnine::fill::ones,_dev.id());}

    static SO3partArray gaussian(const Gdims& _adims, const int _l, const int _n, const int _nbu=-1){
      return SO3partArray(_adims,_l,_n,_nbu,cnine::fill::gaussian);}
    static SO3partArray gaussian(const Gdims& _adims, const int _l, const int _n, const int _nbu, const device& _dev){
      return SO3partArray(_adims,_l,_n,_nbu,cnine::fill::gaussian,_dev.id());}

    SO3partArray(const Gdims& _adims, const int _l, const int _n, 
      std::function<complex<float>(const Gindex& ix, const int i, const int m)> fn, const device& _dev=0):
      SO3partArray(_adims,_l,_n,cnine::fill::raw){
      for(int j=0; j<aasize; j++){
	SO3part P=cell(j);
	for(int i=0; i<n; i++)
	  for(int m=0; m<2*l+1; m++)
	    P.set_value(i,m,fn(Gindex(j,adims),i,m));
      }
      to_device(_dev.id());
    }


   public: // ---- Copying ------------------------------------------------------------------------------------
    

    SO3partArray(const SO3partArray& x):
      GELIB_SO3PARTARRAY_IMPL(x){}
      
    SO3partArray(SO3partArray&& x):
      GELIB_SO3PARTARRAY_IMPL(std::move(x)){}

    SO3partArray& operator=(const SO3partArray& x){
      GELIB_SO3PARTARRAY_IMPL::operator=(x);
      return *this;
    }

    SO3partArray& operator=(SO3partArray&& x){
      GELIB_SO3PARTARRAY_IMPL::operator=(std::move(x));
      return *this;
    }


  public: // ---- Variants -----------------------------------------------------------------------------------


    SO3partArray(const SO3partArray& x, const int _dev):
      GELIB_SO3PARTARRAY_IMPL(x,_dev){}
      
    SO3partArray(const SO3partArray& x, const cnine::device& _dev):
      GELIB_SO3PARTARRAY_IMPL(x,_dev.id()){}
      
    SO3partArray(SO3partArray& x, const cnine::view_flag& flag):
      GELIB_SO3PARTARRAY_IMPL(x,flag){}
      
    template<typename FILLTYPE, typename = 
	     typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partArray(const SO3partArray& x, const FILLTYPE& fill):
      SO3partArray(x.adims,x.getl(),x.getn(),x.get_nbu(),fill,x.dev){}

    template<typename FILLTYPE, typename = 
	     typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partArray(const SO3partArray& x, const Gdims& _adims, const FILLTYPE& fill):
      SO3partArray(_adims,x.getl(),x.getn(),x.get_nbu(),fill,x.dev){}

    template<typename FILLTYPE, typename = 
	     typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3partArray(const SO3part& x, const Gdims& _adims, const FILLTYPE& fill):
      SO3partArray(_adims,x.getl(),x.getn(),x.get_nbu(),fill,x.dev){}


  public: // ---- Transport ----------------------------------------------------------------------------------


    SO3partArray& move_to(const device& _dev){
      GELIB_SO3PARTARRAY_IMPL::move_to_device(_dev.id());
      return *this;
    }
    
    SO3partArray& move_to_device(const int _dev){
      GELIB_SO3PARTARRAY_IMPL::move_to_device(_dev);
      return *this;
    }
    
    SO3partArray to(const device& _dev) const{
      return SO3partArray(*this,_dev.id());
    }

    SO3partArray to_device(const int _dev) const{
      return SO3partArray(*this,_dev);
    }

    
  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3partArray(const GELIB_SO3PARTARRAY_IMPL& x):
      GELIB_SO3PARTARRAY_IMPL(x){}
      
    SO3partArray(GELIB_SO3PARTARRAY_IMPL&& x):
      GELIB_SO3PARTARRAY_IMPL(std::move(x)){}

    //SO3partArray to(const device& _dev) const{
    //return SO3partArray(*this,_dev);
    //}

    explicit operator SO3part() const{
      return get_cell(0);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    SO3partArray shape(const Gdims& _adims) const{
      return GELIB_SO3PARTARRAY_IMPL::arrayshape(_adims);
    }
     
    SO3partArray reshape(const Gdims& _adims){
      return GELIB_SO3PARTARRAY_IMPL::change_arrayshape(_adims);
    }
     
    SO3partArray as_shape(const Gdims& _adims) const{
      return GELIB_SO3PARTARRAY_IMPL::as_arrayshape(_adims);
    }
     
 
  public: // ---- Broadcasting and reductions ----------------------------------------------------------------


    SO3partArray(const Gdims& _adims, const SO3part& x, const device& _dev=0):
      SO3partArray(_adims,x.getl(),x.getn(),x.get_nbu(),cnine::fill::raw,x.dev){
      broadcast_copy(x);
      to_device(_dev.id());
    }

    SO3partArray broaden(const int ix, const int n) const{
      assert(ix<=adims.size());
      SO3partArray R(*this, adims.insert(ix,n),cnine::fill::raw);
      R.broadcast_copy(ix,*this);
      return R;
    }

    SO3partArray reduce(const int ix) const{
      assert(ix<adims.size());
      SO3partArray R(*this, adims.remove(ix),cnine::fill::zero);
      R.add_reduce(*this,ix);
      return R;
    }

    SO3part reduce() const{
      SO3partArray A=as_shape(Gdims(aasize)).reduce(0);
      SO3part R=A.get_cell(Gindex(0));
      A.is_view=true;
      R.is_view=false;
      return R;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    //SO3partArray apply(std::function<complex<float>(const complex<float>)> fn){
    //return Obj(Cengine_engine->push<ctensor_apply_op>(hdl,[]fn),dims,nbu);
    //}

    //SO3partArray apply(std::function<complex<float>(const int i, const int m, const complex<float>)> fn){
    //return GELIB_SO3PARTARRAY_IMPL(Cengine_engine->push<ctensor_apply2_op>
    //(hdl,[&](const int i, const int j, complex<float> x){return fn(j,i-l,x);}),dims,nbu);
    //};
  


  public: // ---- Accumulating operations ------------------------------------------------------------------


    //void add_prod(const ctensor& M, const SO3partArray& p){
    //GELIB_SO3PARTARRAY_IMPL::add_Mprod_AT<0>(p,M);
    //}

    void add_prod_back0_into(GELIB_SO3PARTARRAY_IMPL& Mg, const SO3partArray& p) const{
      Mg.add_Mprod_TA<2>(*this,p);
    }

    void add_prod_back1_into(const GELIB_SO3PARTARRAY_IMPL& M, SO3partArray& pg) const{
      pg.add_Mprod_AA<2>(*this,M);
    }

    void add_norm2_back(const cscalar& g, const SO3partArray& x){
      add(x,g);
      add_conj(x,g.val);
    }


  public: // ---- Spherical harmonics -----------------------------------------------------------------------

    /*
    static SO3partArray  spharm(const int l, const int n, const Gtensor<float>& _x, const int _nbu=-1, const device& dev=0){
      assert(_x.k==1);
      assert(_x.dims[0]==3);
      float x=_x(0);
      float y=_x(1);
      float z=_x(2);
      //return SO3partArray(engine::new_SO3partArray_spharm(l,x(0),x(1),x(2),_nbu,dev.id()),l,1,_nbu);

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

      return SO3partArray();
      //return newSO3partArray(engine::new_SO3partArray(R),l,1);
      //return SO3partArray(Cengine_engine->push<new_SO3partArray_from_Gtensor_op>(R,dev.id()),l,n);
    }
    */
    
    /*
    SO3partArray static spharm(const int l, const int n, const float x, const float y, const float z, const int _nbu=-1, const device_id& dev=0){
      //return SO3partArray(engine::new_SO3partArray_spharm(l,x,y,z,_nbu,dev.id()),l,1,_nbu);
      return SO3partArray(Cengine_engine->push<new_spharm_op>(l,n,x,y,z,_nbu,dev.id()),l,n);
    }
    */
 



  public: // ---- Not in-place operations --------------------------------------------------------------------

    
    //SO3partArray plus(const SO3partArray& x){
    //return SO3partArray(Cengine_engine->push<ctensor_add_op>(hdl,x.hdl,dims),l,n,nbu);
    //}

    //SO3partArray normalize() const{
      //return SO3partArray(::Cengine::engine::ctensor_normalize_cols(hdl),l,n,nbu);
    //}

    //void add_normalize_back(const SO3partArray& g, const SO3partArray& x){
      //replace(hdl,::Cengine::engine::ctensor_add_normalize_cols_back(hdl,g.hdl,x.hdl));
    //}

    //SO3partArray chunk(const int i, const int n=1) const {
    //return GELIB_SO3PARTARRAY_IMPL::chunk(1,i,n);
    //}

    SO3partArray rotate(const SO3element& r){ // TODO 
      cnine::CtensorObj D(WignerMatrix<float>(l,r),dev);
      cnine::CtensorArray Darr(adims,D);
      SO3partArray R(adims,l,n,cnine::fill::zero,dev);
      R.add_Mprod_AA<0>(Darr,*this);
      return R;
    }

    /*
    GELIB_SO3PARTARRAY_IMPL fragment_norms() const{
      return column_norms();
    }

    SO3partArray divide_fragments(const GELIB_SO3PARTARRAY_IMPL& N) const{
      return divide_columns(N);
    }
    */


  public: // ---- In-place cellwise arithmetic ---------------------------------------------------------------


    SO3partArray& operator+=(const SO3partArray& y){
      add(y);
      return *this;
    }

    SO3partArray& operator-=(const SO3partArray& y){
      subtract(y);
      return *this;
    }

 
  public: // ---- Cellwise arithmetic -------------------------------------------------------------------------


    SO3partArray operator+(const SO3partArray& y) const{
      SO3partArray R(*this);
      R.add(y);
      return R;
    }

    SO3partArray operator-(const SO3partArray& y) const{
      SO3partArray R(*this);
      R.subtract(y);
      return R;
    }

    /*
    SO3partArray operator*(const CtensorArray& M) const{
      SO3partArray R(adims,l,M.get_dims()[0],nbu,cnine::fill::zero);
      R.add_Mprod_AT<0>(*this,M);
      return R;
    }
    
    SO3partArray operator*(const Transpose<CtensorArray>& _M) const{
      const CtensorArray& M=_M.obj;
      assert(M.k==2);
      SO3partArray R(adims,l,M.get_dims()[1],nbu,cnine::fill::zero);
      R.add_Mprod_AA<0>(*this,M);
      return R;
    }
    */
    
    CtensorArray operator*(const Transpose<SO3partArray>& _y) const{
      const SO3partArray& y=_y.obj;
      assert(y.getl()==getl());
      CtensorArray R(adims,Gdims(getn(),y.getn()),cnine::fill::zero);
      R.add_mprod_TA(y,*this);
      return R;
    }


  public: // ---- Inplace broadcasting arithmetic ------------------------------------------------------------


    SO3partArray& operator*=(const complex<float> c){
      inplace_times(c);
      return *this;
    }

    SO3partArray& operator*=(const cscalar& c){
      inplace_times(c);
      return *this;
    }

    SO3partArray& operator/=(const complex<float> c){
      inplace_times(complex<float>(1.0)/c);
      return *this;
    }

    SO3partArray& operator/=(const cscalar& c){
      inplace_div(c);
      return *this;
    }


  public: // ---- Broadcasting arithmetic --------------------------------------------------------------------


    SO3partArray operator*(const complex<float> c) const{
      SO3partArray R(adims,l,n,nbu,cnine::fill::zero,dev);
      R.add(*this,c);
      return R;
    }

    SO3partArray operator*(const cscalar& c) const{
      SO3partArray R(adims,l,n,nbu,cnine::fill::zero,dev);
      R.add(*this,c);
      return R;
    }

    SO3partArray operator/(const complex<float> c) const{
      SO3partArray R(adims,l,n,nbu,cnine::fill::zero,dev);
      R.add_divide(*this,c);
      return R;
    }

    SO3partArray operator/(const cscalar& c) const{
      SO3partArray R(adims,l,n,nbu,cnine::fill::zero,dev);
      R.add_divide(*this,c);
      return R;
    }

    SO3partArray operator+(const SO3part& y) const{
      SO3partArray R(*this);
      R.add_broadcast(y);
      return R;
    }
    
    SO3partArray operator-(const SO3part& y) const{
      SO3partArray R(*this);
      R.subtract_broadcast(y);
      return R;
    }


  public: // ---- Scattering arithmetic --------------------------------------------------------------------
    

    SO3partArray& operator*=(const Scatter<Ctensor>& C){
      inplace_scatter_times(C.obj);
      return *this;
    }
      
    SO3partArray operator*(const Scatter<Ctensor>& C) const{
      SO3partArray R(*this);
      R.inplace_scatter_times(C.obj);
      return R;
    }
      
    SO3partArray& operator/=(const Scatter<Ctensor>& C){
      inplace_scatter_div(C.obj);
      return *this;
    }
      
    SO3partArray operator/(const Scatter<Ctensor>& C) const{
      SO3partArray R(*this);
      R.inplace_scatter_div(C.obj);
      return R;
    }
      

  public: // ---- I/O --------------------------------------------------------------------------------------


    string classname() const{
      return "GEnet::SO3partArray";
    }

    string describe() const{
      ostringstream oss;
      oss<<"SO3partArray(l="<<to_string(l)<<",n="<<to_string(n)<<"):"<<endl;
      oss<<"adims="<<adims<<" "<<astrides<<endl;
      oss<<"cdims="<<cdims<<" "<<cstrides<<endl;
      oss<<"dims="<<dims<<" "<<strides<<endl;
      return oss.str();
    }

    string repr(const string indent="") const{
      return indent+"GElib::SO3partArray"+adims.str()+" l="+to_string(getl())+",n="+to_string(getn())+")";
    }

    friend ostream& operator<<(ostream& stream, const SO3partArray& x){
      stream<<x.str(); return stream;}

  };



  // ---------------------------------------------------------------------------------------------------------
  // ---- Post-class functions -------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------------------


  /* provided in OperationTemplates.hpp
  inline SO3partArray operator*(const cnine::CscalarObj& c, const SO3partArray& x){
    return x*c;
  }
  */ 

  inline SO3partArray operator*(const cnine::CtensorArray& M, const SO3partArray& x){
    SO3partArray R(x.adims,x.getl(),M.get_cdim(0),x.get_nbu(),cnine::fill::zero);
    R.add_Mprod_AT<0>(x,M);
    return R;
  }

  inline SO3partArray operator*(const cnine::Transpose<cnine::CtensorArray>& _M, const SO3partArray& x){
    const cnine::CtensorArray& M=_M.obj;
    assert(M.k==2);
    SO3partArray R(x.adims,x.getl(),M.get_cdim(1),x.get_nbu(),cnine::fill::zero);
    R.add_Mprod_AA<0>(x,M);
    return R;
  }

  inline SO3partArray operator*(const cnine::CtensorObj& M, const SO3partArray& x){
    SO3partArray R(x.adims,x.getl(),M.get_dim(0),x.get_nbu(),cnine::fill::zero);
    R.broadcast_add_Mprod_AT<0>(x,M);
    return R;
  }

  inline SO3partArray operator*(const cnine::Transpose<cnine::CtensorObj>& _M, const SO3partArray& x){
    const cnine::CtensorObj& M=_M.obj;
    SO3partArray R(x.adims,x.getl(),M.get_dim(0),x.get_nbu(),cnine::fill::zero);
    R.broadcast_add_Mprod_AA<0>(x,M);
    return R;
  }

  inline SO3partArray operator+(const SO3part& x, const SO3partArray& y){
    return y+x;
  }

  inline SO3partArray operator-(const SO3part& x, const SO3partArray& y){
    SO3partArray R(y,cnine::fill::zero);
    R.add_broadcast(x);
    R-=y;
    return R;
  }

  inline SO3partArray operator*(const cnine::Scatter<cnine::CtensorObj>& C, const SO3partArray& x){
    return x*C;
  }


  // ---- CGproduct ------------------------------------------------------------------------------------------

  
  inline SO3partArray CGproduct(const SO3partArray& x, const SO3partArray& y, const int l){
    assert(x.adims==y.adims);
    assert(x.dev==y.dev);
    SO3partArray R(x.adims,l,x.getn()*y.getn(),cnine::fill::zero,x.dev);
    R.add_CGproduct(x,y,0);
    return R;
  }

  /*
  inline SO3partArray CGproduct(const cnine::Outer<SO3partArray,SO3partArray>& args, const int l){
    const SO3partArray& x=args.obj0;
    const SO3partArray& y=args.obj1;
    assert(x.dev==y.dev);
    SO3partArray R(cnine::Gdims(x.adims,y.adims),l,x.getn()*y.getn(),cnine::fill::zero,x.dev);
    R.add_outer_CGproduct(x,y,0);
    return R;
  }

  inline SO3partArray CGproduct(const cnine::Convolve<SO3partArray,SO3partArray>& args, const int l){
    const SO3partArray& x=args.obj0;
    const SO3partArray& y=args.obj1;
    assert(x.dev==y.dev);
    SO3partArray R(x.adims.convolve(y.adims),l,x.getn()*y.getn(),cnine::fill::zero,x.dev);
    R.add_convolve_CGproduct(x,y,0);
    return R;
  }

  inline SO3partArray CGproduct(const SO3part& x, const SO3partArray& y, const int l){
    assert(x.dev==y.dev);
    SO3partArray R(y.adims,l,x.getn()*y.getn(),cnine::fill::zero,x.dev);
    R.add_CGproduct(x,y,0);
    return R;
  }

  inline SO3partArray CGproduct(const SO3partArray& x, const SO3part& y, const int l){
    assert(x.dev==y.dev);
    SO3partArray R(x.adims,l,x.getn()*y.getn(),cnine::fill::zero,x.dev);
    R.add_CGproduct(x,y,0);
    return R;
  }

  inline SO3partArray CGproduct_back0(const SO3partArray& g, const SO3partArray& y, const int l){
    assert(g.adims==y.adims);
    assert(g.dev==y.dev);
    SO3partArray R(y.adims,l,g.getn()/y.getn(),cnine::fill::zero,y.dev);
    R.add_CGproduct_back0(g,y,0);
    return R;
  }

  inline SO3partArray CGproduct_back1(const SO3partArray& g, const SO3partArray& x, const int l){
    assert(g.adims==x.adims);
    assert(g.dev==x.dev);
    SO3partArray R(x.adims,l,g.getn()/x.getn(),cnine::fill::zero,x.dev);
    R.add_CGproduct_back1(g,x,0);
    return R;
  }
  */


}

#endif 

    /* redundant 
    int get_dev() const{
      return dev;
    }
    
    int getl() const{
      return l;
    }

    int getn() const{
      return n;
    }

    int get_nbu() const{ 
      return nbu;
    }
    */


    /*
    void add_to_frags(const int offs, const SO3partArray& x){
      GELIB_SO3PARTARRAY_IMPL::add_to_chunk(1,offs,x);
    }

    void add_frags_of(const SO3partArray& x, const int offs, const int n){
      GELIB_SO3PARTARRAY_IMPL::add_chunk_of(x,1,offs,n);
    }
    */
    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //SO3partArray(const Gdims& _adims, const SO3part_spec& _spec, const FILLTYPE& fill):
    //SO3partArray(_adims,_spec.l,_spec.n,_spec.nbu,fill,_spec.dev){}

