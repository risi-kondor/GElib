
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
#include "SO3partA.hpp"
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
      return SO3part(_l,_n,-1,cnine::fill::raw);}
    static SO3part raw(const int _l, const int _n, const int _nbu){
      return SO3part(_l,_n,_nbu,cnine::fill::raw);}
    static SO3part raw(const int _l, const int _n, const int _nbu, const device& _dev){
      return SO3part(_l,_n,_nbu,cnine::fill::raw,_dev);}

    static SO3part zero(const int _l, const int _n){
      return SO3part(_l,_n,-1,cnine::fill::zero);}
    static SO3part zero(const int _l, const int _n, const int _nbu){
      return SO3part(_l,_n,_nbu,cnine::fill::zero);}
    static SO3part zero(const int _l, const int _n, const int _nbu, const device& _dev){
      return SO3part(_l,_n,_nbu,cnine::fill::zero,_dev);}

    static SO3part ones(const int _l, const int _n){
      return SO3part(_l,_n,-1,cnine::fill::ones);}
    static SO3part ones(const int _l, const int _n, const int _nbu){
      return SO3part(_l,_n,_nbu,cnine::fill::ones);}
    static SO3part ones(const int _l, const int _n, const int _nbu, const device& _dev){
      return SO3part(_l,_n,_nbu,cnine::fill::ones,_dev);}

    static SO3part gaussian(const int _l, const int _n){
      return SO3part(_l,_n,-1,cnine::fill::gaussian);}
    static SO3part gaussian(const int _l, const int _n, const int _nbu){
      return SO3part(_l,_n,_nbu,cnine::fill::gaussian);}
    static SO3part gaussian(const int _l, const int _n, const int _nbu, const device& _dev){
      return SO3part(_l,_n,_nbu,cnine::fill::gaussian,_dev.id());}

    static SO3part spharm(const int _l, const float x, const float y, const float z){
      SO3part R(_l,1,-1,cnine::fill::zero);
      R.add_spharm(x,y,z);
      return R;
    }


    SO3part(const int _l, const int _n, 
      std::function<complex<float>(const int i, const int m)> fn):
      GELIB_SO3PART_IMPL({2*_l+1,_n},[&](const int i, const int j){return fn(j,i-_l);}){}
    

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


  public: // ---- Access -------------------------------------------------------------------------------------


  public: // ---- Operations ---------------------------------------------------------------------------------


  public: // ---- Accumulating operations ------------------------------------------------------------------


    void add_prod(const ctensor& M, const SO3part& p){
      GELIB_SO3PART_IMPL::add_Mprod_AT<0>(p,M);
    }

    void add_prod_back0_into(GELIB_SO3PART_IMPL& Mg, const SO3part& p) const{
      Mg.add_Mprod_TA<2>(*this,p);
    }

    void add_prod_back1_into(const GELIB_SO3PART_IMPL& M, SO3part& pg) const{
      pg.add_Mprod_AA<2>(*this,M);
    }

    void add_norm2_back(const cscalar& g, const SO3part& x){
      add(x,g);
      add_conj(x,g.val);
    }

    void add_to_frags(const int offs, const SO3part& x){
      GELIB_SO3PART_IMPL::add_to_chunk(1,offs,x);
    }

    void add_frags_of(const SO3part& x, const int offs, const int n){
      GELIB_SO3PART_IMPL::add_chunk_of(x,1,offs,n);
    }


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


    SO3part& operator+=(const SO3part& y){
      add(y);
      return *this;
    }

    SO3part& operator-=(const SO3part& y){
      subtract(y);
      return *this;
    }

    /*
    SO3part& normalize_fragments(){
      //GELIB_SO3PART_IMPL a=column_norms();
      //divide_columns(column_norms());
      (*this)=(divide_columns(column_norms()));
      return *this;
    }
    */


  public: // ---- Binary operators ---------------------------------------------------------------------------


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


  inline cnine::CscalarObj norm2(const SO3part& x){
    cnine::CscalarObj r(x.get_nbu(),cnine::fill::zero);
    x.add_norm2_into(r);
    return r;
  }

  inline cnine::CscalarObj inp(const SO3part& x, const SO3part& y){
    cnine::CscalarObj r(x.get_nbu(),cnine::fill::zero);
    x.add_inp_into(r,y);
    return r;
  }

  inline SO3part operator*(const cnine::CscalarObj& c, const SO3part& x){
    return x*c; 
  }

  inline SO3part operator*(const cnine::CtensorObj& M, const SO3part& x){
    SO3part R(x.getl(),M.get_dims()[0],cnine::fill::zero);
    R.add_Mprod_AT<0>(x,M);
    return R;
  }

  inline SO3part CGproduct(const SO3part& x, const SO3part& y, const int l){
    const int dev=x.get_dev()*y.get_dev();
    SO3part R(l,x.getn()*y.getn(),cnine::fill::zero,dev);
    R.add_CGproduct(x,y,0);
    return R;
  }

  
  inline SO3part DiagCGproduct(const SO3part& x, const SO3part& y, const int l){
    assert(x.getn()==y.getn());
    const int dev=x.get_dev()*y.get_dev();
    SO3part R(l,x.getn(),cnine::fill::zero,dev);
    R.add_DiagCGproduct(x,y,0);
    return R;
  }

  
  /*
  ostream& operator<<(ostream& stream, const MemberExpr2<SO3part,CscalarObj,complex<float>,int>& x){
    stream<<x.str(); return stream;
  }

  ostream& operator<<(ostream& stream, const ConstSO3partElement& x){
    stream<<x.str(); return stream;
  }
  */


  // ---- Downcasting ----------------------------------------------------------------------------------------
  

  /*
  inline SO3part& asSO3part(Dnode* x){
    if(!dynamic_cast<SO3part*>(x->obj)){
      if(!x->obj) cerr<<"GEnet error: Dobject does not exist."<<endl;
      else {cerr<<"GEnet error: Dobject is of type "<<x->obj->classname()<<" instead of SO3part."<<endl;}
    }
    assert(dynamic_cast<SO3part*>(x->obj));
    return *static_cast<SO3part*>(x->obj);
  }

  inline const SO3part& asSO3part(const Dnode* x){
    if(!dynamic_cast<const SO3part*>(x->obj)){
      if(!x->obj) cerr<<"GEnet error: Dobject does not exist."<<endl;
      else {cerr<<"GEnet error: Dobject is of type "<<x->obj->classname()<<" instead of SO3part."<<endl;}
    }
    assert(dynamic_cast<const SO3part*>(x->obj));
    return *static_cast<const SO3part*>(x->obj);
  }

  inline SO3part& asSO3part(Dobject* x){
    if(!dynamic_cast<SO3part*>(x)){
      if(!x) cerr<<"GEnet error: Dobject does not exist."<<endl;
      else {cerr<<"GEnet error: Dobject is of type "<<x->classname()<<" instead of SO3part."<<endl;}
    }
    assert(dynamic_cast<SO3part*>(x));
    return *static_cast<SO3part*>(x);
  }

  inline const SO3part& asSO3part(const Dobject* x){
    if(!dynamic_cast<const SO3part*>(x)){
      if(!x) cerr<<"GEnet error: Dobject does not exist."<<endl;
      else {cerr<<"GEnet error: Dobject is of type "<<x->classname()<<" instead of SO3part."<<endl;}
    }
    assert(dynamic_cast<const SO3part*>(x));
    return *static_cast<const SO3part*>(x);
  }

  inline SO3part& asSO3part(Dobject& x){
    if(!dynamic_cast<SO3part*>(&x))
      cerr<<"GEnet error: Dobject is of type "<<x.classname()<<" instead of SO3part."<<endl;
    assert(dynamic_cast<SO3part*>(&x));
    return static_cast<SO3part&>(x);
  }

  inline const SO3part& asSO3part(const Dobject& x){
    if(!dynamic_cast<const SO3part*>(&x))
      cerr<<"GEnet error: Dobject is of type "<<x.classname()<<" instead of SO3part."<<endl;
    assert(dynamic_cast<const SO3part*>(&x));
    return static_cast<const SO3part&>(x);
  }
  */


}

#endif 

    //hdl=engine::new_ctensor({2*l+1,n},-1,device);
    //hdl=engine::new_ctensor({2*l+1,n},-1,device);
    //hdl=engine::new_ctensor_zero({2*l+1,n},-1,device);
    //hdl=engine::new_ctensor_gaussian({2*l+1,n},-1,device);
    //hdl(engine::ctensor_copy(x.hdl)){}
  // ---- SO3partElement

  /*
  inline SO3partElement::operator CscalarObj() const{
    return obj.get(i,m);
  }

  inline SO3partElement& SO3partElement::operator=(const CscalarObj& x){
    obj.set(i,m,x);    
    return *this;
  }

  inline complex<float> SO3partElement::get_value() const{
    return obj.get_value(i,m);
  }
  
  inline SO3partElement& SO3partElement::set_value(const complex<float> x){
    obj.set_value(i,m,x);
    return *this;
  }
  */

  /*
  inline ConstSO3partElement::operator CscalarObj() const{
    return obj.get(i,m);
  }

  inline complex<float> ConstSO3partElement::get_value() const{
    return obj.get_value(i,m);
  }
  */
    /*
    SO3part(const int _l, const int _n, const fill_zero& fill, const int device=0):
      GELIB_SO3PART_IMPL({2*_l+1,_n},-1,fill,device), l(_l), n(_n){}

    SO3part(const int _l, const int _n, const fill_ones& fill, const int device=0):
      GELIB_SO3PART_IMPL({2*_l+1,_n},-1,fill,device), l(_l), n(_n){}

    SO3part(const int _l, const int _n, const fill_gaussian& fill, const int device=0):
      GELIB_SO3PART_IMPL({2*_l+1,_n},-1,fill,device), l(_l), n(_n){}
    */

    /*
    SO3part(const int _l, const int _n, const int _nbu, const fill_zero& fill, const int device=0):
      CtensorObj({2*_l+1,_n},_nbu,fill,device), l(_l), n(_n){}

    SO3part(const int _l, const int _n, const int _nbu, const fill_ones& fill, const int device=0):
      CtensorObj({2*_l+1,_n},_nbu,fill,device), l(_l), n(_n){}

    SO3part(const int _l, const int _n, const int _nbu, const fill_gaussian& fill, const int device=0):
      CtensorObj({2*_l+1,_n},_nbu,fill,device), l(_l), n(_n){}
    */

    //SO3part(const int _l, const int _n):
    //CtensorObj({2*_l+1,_n},-1), l(_l), n(_n){}

    //SO3partSeed* seed() const{
    //return new SO3partSeed(l,n,nbu);
    //}

    //void add_CGproduct(const SO3part& x, const SO3part& y, const int offs=0){
      // replace(hdl,Cengine_engine->push<SO3part_add_CGproduct_op>(hdl,x.hdl,y.hdl,dims,x.dims,y.dims,offs));
    //}
    
    //void add_CGproductBack0(const SO3part& g, const SO3part& y, const int offs=0){
    //GELIB_UNIMPL();
      //replace(hdl,Cengine_engine->push<SO3part_add_CGproduct_back0_op>(hdl,g.hdl,y.hdl,dims,g.dims,y.dims,offs));
    //}

    //void add_CGproductBack1(const SO3part& g, const SO3part& x, const int offs=0){
    //GELIB_UNIMPL();
      //replace(hdl,Cengine_engine->push<SO3part_add_CGproduct_back1_op>(hdl,g.hdl,x.hdl,dims,g.dims,x.dims,offs));
    //}
    /*
    SO3part(const vector<const SO3part*> v):
      GELIB_SO3PART_IMPL(fill::cat,1,
	::Cengine::apply<const SO3part*, const CtensorObj*>(v,[](const SO3part* x){return x;})), 
      l(v[0]->l){
      n=dims[1];
    }
    */
    /*
    Dobject* clone() const{
      return new SO3part(*this);
    }

    Dobject* spawn(const fill_zero& fill) const{
      return new SO3part(l,n,nbu,fill::zero,dev);
    }

    Dobject* spawn(const fill_zero& fill, const int dev) const{
      return new SO3part(l,n,nbu,fill::zero,dev);
    }

    Dobject* spawn(const fill_gaussian& fill) const{
      return new SO3part(l,n,nbu,fill::gaussian);
    }

    Dobject* spawn(const int _l, const int _n, const fill_zero& fill) const{
      return new SO3part(_l,_n,nbu,fill::zero);
    }
    */
    
    /*
    void add_sum(const vector<SO3part*> v){
      vector<Chandle*> h(v.size());
      for(int i=0; i<v.size(); i++) h[i]=v[i]->hdl;
      replace(hdl,Cengine_engine->push<ctensor_add_sum_op>(hdl,h));
    }
    */



    //SO3part static spharm(const int l, const GELIB_SO3PART_IMPL& x, const int _nbu=-1, const device_id& dev=0){
    //return SO3part(::Cengine::engine::new_SO3part_spharm(l,x,_nbu,dev),l,1,nbu);
    //}

    //int get_dev() const{
    //return dev;
    //}
    
    //int get_nbu() const{ 
    //return nbu;
    //}

    /*
    complex<float> get_value(const int i, const int m) const{
      return GELIB_SO3PART_IMPL::get(i,m+l);
    }

    SO3part& set_value(const int i, const int m, complex<float> x){
      GELIB_SO3PART_IMPL::set(m+l,i,x);
      return *this; 
    }

    SO3part& set(const int i, const int m, complex<float> x){
      GELIB_SO3PART_IMPL::set(m+l,i,x);
      return *this; 
    }

    cscalar get(const int i, const int m) const{
      return GELIB_SO3PART_IMPL::get(i,m+l);
    }

    SO3part& set(const int i, const int m, const cscalar& x){
      GELIB_SO3PART_IMPL::set(i,m+l,x.val);
      return *this; 
    }
    */


  /*
  inline SO3part SO3partSeed::spawn(const fill_zero& fill){
    //cout<<"No seed!"<<endl;
    //exit(0);
    return SO3part(l,n,nbu,fill::zero,dev);
  }
  */

  /*
  inline SO3fragExpr& SO3fragExpr::operator=(const SO3fragExpr& x){
    operator=(SO3part(x));
    return *this;
  };

  inline SO3fragExpr::operator SO3part() const{
    return owner->chunk(1,i,n);
  }

  inline SO3fragExpr& SO3fragExpr::operator=(const SO3part& x){
    GENET_CHECK_EQ(n,x.getn(),"SO3fragExpre::operator=","number of fragments does not match.");
    owner->set_chunk(x,1,i);
    return *this;
  }

  ostream& operator<<(ostream& stream, const SO3fragExpr& x){
    stream<<SO3part(x).str(); return stream;
  }
  */
    /*
    SO3part(const int _l, const int _n, const int _nbu=-1, const int _dev=0):
      GELIB_SO3PART_IMPL({2*_l+1,_n},_nbu,cnine::fill_raw(),_dev){}//, l(_l), n(_n){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3part(const int _l, const int _n, const FILLTYPE& fill, const int _dev=0):
      GELIB_SO3PART_IMPL({2*_l+1,_n},fill,_dev){}//, l(_l), n(_n){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3part(const int _l, const int _n, const int _nbu, const FILLTYPE& fill, const int _dev=0):
      GELIB_SO3PART_IMPL({2*_l+1,_n},_nbu,fill,_dev){}//, l(_l), n(_n){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3part(const int _l, const int _n, const FILLTYPE& fill, const device& _dev):
      SO3part(_l,_n,-1,fill,_dev.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3part(const int _l, const int _n, const int _nbu, const FILLTYPE& fill, 
      const device& _dev):
      SO3part(_l,_n,_nbu,fill,_dev.id()){}
    */

   // ---- Member expressions


    //MemberExpr2<SO3part,cscalar,complex<float>,int,int> operator()(const int i, const int m){
    //return MemberExpr2<SO3part,cscalar,complex<float>,int,int>(*this,i,m);
    //}

    //constMemberExpr2<SO3part,cscalar,complex<float>,int,int> operator()(const int i, const int m) const{
    //return constMemberExpr2<SO3part,cscalar,complex<float>,int,int>(*this,i,m);
    //}

    //SO3partElement operator()(const int i, const int m){
    //return SO3partElement(*this,i,m);
    //}

    //ConstSO3partElement operator()(const int i, const int m) const{
    //return ConstSO3partElement(*this,i,m);
    //}

    //SO3fragExpr fragment(const int i){
    //return SO3fragExpr(this,i);
    //}


    //public: // shorthands 

    //complex<float> value(const int i, const int m) const {return get_value(i,m);}
    //SO3part& set(const int i, const int m, complex<float> x) {return set_value(i,m,x);}
      


