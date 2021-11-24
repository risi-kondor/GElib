
// This file is part of GElib, a C++/CUDA library for group
// equivariant tensor operations. 
// 
// Copyright (c) 2021, Imre Risi Kondor and Erik H Thiede
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef _GElibSO3vecArray
#define _GElibSO3vecArray

#include "GElib_base.hpp"
#include "ArrayPack.hpp"
#include "SO3vec.hpp"
#include "SO3partArray.hpp"
#include "SO3element.hpp"
#include "WignerMatrix.hpp"
#include "CscalarObj.hpp"
#include "CtensorObj.hpp"
#include "CtensorArray.hpp"


namespace GElib{


  class SO3vecArray: public cnine::ArrayPack<SO3partArray>{
  public:


    typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    template<typename TYPE>
    using Transpose=cnine::Transpose<TYPE>;
    //template<typename OBJ0, typename OBJ1>
    //using Outer=cnine::Outer;


    template<typename TYPE>
    using Gtensor=cnine::Gtensor<TYPE>;

    typedef cnine::Gdims Gdims;
    typedef cnine::Gindex Gindex;
    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;
    typedef cnine::CtensorArray ctensor_arr;

    using cnine::ArrayPack<SO3partArray>::array; 

    SO3type tau; 
    //int nbu; 
    //int dev=0; 

    //vector<SO3partArray*> parts;

    ~SO3vecArray(){
      //for(auto p:parts)
      //delete p;
    }


    // ---- Constructors -------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecArray(const Gdims& _adims, const SO3type& _tau, const FILLTYPE& fill, const int _dev=0):
      ArrayPack(_adims), tau(_tau){
      for(int l=0; l<tau.size(); l++) 
	array.push_back(new SO3partArray(_adims,l,tau[l],-1,fill,_dev));
    }

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecArray(const Gdims& _adims, const SO3type& _tau, const int _nbu, const FILLTYPE& fill, const int _dev=0):
      ArrayPack(_adims), tau(_tau){
      for(int l=0; l<tau.size(); l++) 
	array.push_back(new SO3partArray(_adims,l,tau[l],_nbu,fill,_dev));
    }

    static SO3vecArray zero(const Gdims& _adims, const SO3type& _tau, const int _nbu=-1){
      return SO3vecArray(_adims,_tau,_nbu,cnine::fill::zero);}
    static SO3vecArray zero(const Gdims& _adims, const SO3type& _tau, const int _nbu, const device& _dev){
      return SO3vecArray(_adims,_tau,_nbu,cnine::fill::zero,_dev.id());}

    static SO3vecArray ones(const Gdims& _adims, const SO3type& _tau, const int _nbu=-1){
      return SO3vecArray(_adims,_tau,_nbu,cnine::fill::ones);}
    static SO3vecArray ones(const Gdims& _adims, const SO3type& _tau, const int _nbu, const device& _dev){
      return SO3vecArray(_adims,_tau,_nbu,cnine::fill::ones,_dev.id());}

    static SO3vecArray gaussian(const Gdims& _adims, const SO3type& _tau, const int _nbu=-1){
      return SO3vecArray(_adims,_tau,_nbu,cnine::fill::gaussian);}
    static SO3vecArray gaussian(const Gdims& _adims, const SO3type& _tau, const int _nbu, const device& _dev){
      return SO3vecArray(_adims,_tau,_nbu,cnine::fill::gaussian,_dev.id());}

    SO3vecArray(const Gdims& _adims, const SO3type& _tau, 
      std::function<complex<float>(const Gindex& ix, const int l, const int i, const int m)> fn, const device& _dev=0):
      SO3vecArray(_adims,_tau,cnine::fill::raw){
      const int aasize=get_aasize();
      for(int j=0; j<aasize; j++){
	SO3vec v=cell(j);
	for(int l=0; l<v.getL(); l++){
	  SO3part p=v.part(l);
	  const int n=p.getn();
	  for(int i=0; i<n; i++)
	    for(int m=0; m<2*l+1; m++)
	      p.set_value(i,m,fn(Gindex(j,adims),l,i,m));
	}
      }
      to_device(_dev.id());
    }


   public: // ---- Copying ------------------------------------------------------------------------------------
    

    SO3vecArray(const SO3vecArray& x):
      ArrayPack<SO3partArray>(x), tau(x.tau){}
      
    SO3vecArray(SO3vecArray&& x):
      ArrayPack<SO3partArray>(std::move(x)), tau(x.tau){}

    SO3vecArray& operator=(const SO3vecArray& x){
      ArrayPack<SO3partArray>::operator=(x);
      tau=x.tau;
      return *this;
    }

    SO3vecArray& operator=(SO3vecArray&& x){
      ArrayPack<SO3partArray>::operator=(std::move(x));
      tau=x.tau;
      return *this;
    }


   public: // ---- Variants -----------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecArray(const SO3vecArray& x, const FILLTYPE& fill):
      ArrayPack<SO3partArray>(x,fill), tau(x.tau){}
      
    SO3vecArray(const SO3vecArray& x, const int _dev):
      ArrayPack<SO3partArray>(x,_dev), tau(x.tau){}
      
    SO3vecArray(const SO3vecArray& x, const cnine::device& _dev):
      ArrayPack<SO3partArray>(x,_dev.id()), tau(x.tau){}
      
    SO3vecArray(SO3vecArray& x, const cnine::view_flag& flag):
      ArrayPack<SO3partArray>(x,flag), tau(x.tau){}
    
    
  public: // ---- Conversions --------------------------------------------------------------------------------


    SO3vecArray(const ArrayPack<SO3partArray>& x):
      ArrayPack<SO3partArray>(x){
      for(auto p:array)
	tau.push_back(p->getn());
    }

    SO3vecArray(ArrayPack<SO3partArray>&& x):
      ArrayPack<SO3partArray>(std::move(x)){
      for(auto p:array)
	tau.push_back(p->getn());
    }


    SO3vecArray to(const device& _dev) const{
      return SO3vecArray(*this,_dev);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_dev() const{
      if(array.size()==0) return 0;
      return array[0]->get_dev();
    }
    
    SO3type get_tau() const{
      return tau;
    }

    SO3type type() const{
      return tau;
    }

    int size() const{
      return array.size(); 
    }

    int getL() const{
      return tau.size()-1; 
    }

    int get_nbu() const{ 
      if(array.size()==0) return -1;
      return array[0]->get_nbu();
    }

    SO3partArray get_part(const int l) const{
      return get_array(l);
    }

    void set_part(const int l, const SO3partArray& x){
      set_array(l,x);
    }


  public: // ---- Cell Access --------------------------------------------------------------------------------


    SO3vec get_cell(const Gindex& aix) const{
      SO3vec R(cnine::fill_noalloc(), tau, get_nbu(), 0, get_dev());   
      for(auto p:array)
	R.parts.push_back(new SO3part(p->get_cell(aix)));
      return R;
    }

    void copy_cell_into(SO3vec& R, const Gindex& aix) const{
      assert(R.parts.size()==array.size());
      for(int l=0; l<array.size(); l++)
	array[l]->copy_cell_into(*R.parts[l],aix);
    }

    void add_cell_into(SO3vec& R, const Gindex& aix) const{
      assert(R.parts.size()==array.size());
      for(int l=0; l<array.size(); l++)
	array[l]->add_cell_into(*R.parts[l],aix);
    }

    void set_cell(const Gindex& aix, const SO3vec& x) const{
      assert(x.parts.size()==array.size());
      for(int l=0; l<array.size(); l++)
	array[l]->set_cell(aix,*x.parts[l]);
    }

    void add_to_cell(const Gindex& aix, const SO3vec& x) const{
      assert(x.parts.size()==array.size());
      for(int l=0; l<array.size(); l++)
	array[l]->add_to_cell(aix,*x.parts[l]);
    }


    SO3vec cell(const Gindex& aix){
      SO3vec R(cnine::fill_noalloc(), tau, get_nbu(), 0, get_dev());   
      for(auto p:array){
	R.parts.push_back(new SO3part(p->cell(aix)));
	//cout<<R.parts.back()->is_view<<endl;
      }
      return R;
    }


  public: // ---- Broadcasting and reductions ----------------------------------------------------------------


    SO3vecArray(const Gdims& _adims, const SO3vec& x):
      SO3vecArray(_adims,x.get_tau(),x.get_nbu(),cnine::fill::raw,x.dev){
      broadcast_copy(x);
    }

    SO3vecArray repeat(const int ix, const int n) const{
      assert(ix<=adims.size());
      SO3vecArray R(adims.insert(ix,n),get_tau(),get_nbu(),cnine::fill::raw,get_dev());
      R.broadcast_copy(ix,*this);
      return R;
    }

    SO3vecArray reduce(const int ix) const{
      assert(ix<adims.size());
      SO3vecArray R(adims.remove(ix),get_tau(),get_nbu(),cnine::fill::zero,get_dev());
      R.add_reduce(*this,ix);
      return R;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    //SO3vecArray apply(std::function<complex<float>(const complex<float>)> fn){
    //return Obj(Cengine_engine->push<ctensor_apply_op>(hdl,[]fn),dims,nbu);
    //}

    //SO3vecArray apply(std::function<complex<float>(const int i, const int m, const complex<float>)> fn){
    //return GELIB_SO3PARTARRAY_IMPL(Cengine_engine->push<ctensor_apply2_op>
    //(hdl,[&](const int i, const int j, complex<float> x){return fn(j,i-l,x);}),dims,nbu);
    //};
  


  public: // ---- Accumulating operations ------------------------------------------------------------------


  public: // ---- CG-products --------------------------------------------------------------------------------


    template<int selector>
    void add_CGproduct(const SO3vecArray& x, const SO3vecArray& y, const int maxL=-1){
      assert(tau==GElib::CGproduct(x.tau,y.tau,maxL));

      int L1=x.getL(); 
      int L2=y.getL();
      vector<int> offs(tau.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	if(x.tau[l1]==0) continue;
	for(int l2=0; l2<=L2; l2++){
	  if(y.tau[l2]==0) continue;
	  for(int l=std::abs(l2-l1); l<=l1+l2 && (maxL<0 || l<=maxL); l++){
	    if(selector==0) array[l]->add_CGproduct(*x.array[l1],*y.array[l2],offs[l]);
	    //if(selector==1) array[l]->add_outer_CGproduct(*x.array[l1],*y.array[l2],offs[l]);
	    //if(selector==2) array[l]->add_convolve_CGproduct(*x.array[l1],*y.array[l2],offs[l]);
	    offs[l]+=(x.array[l1]->getn())*(y.array[l2]->getn());
	  }
	}
      }
    }

    void add_CGproduct(const SO3vec& x, const SO3vecArray& y, const int maxL=-1){
      assert(tau==GElib::CGproduct(x.tau,y.tau,maxL));

      int L1=x.getL(); 
      int L2=y.getL();
      vector<int> offs(tau.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	if(x.tau[l1]==0) continue;
	for(int l2=0; l2<=L2; l2++){
	  if(y.tau[l2]==0) continue;
	  for(int l=std::abs(l2-l1); l<=l1+l2 && (maxL<0 || l<=maxL); l++){
	    //array[l]->add_CGproduct(*x.parts[l1],*y.array[l2],offs[l]);
	    offs[l]+=(x.parts[l1]->getn())*(y.array[l2]->getn());
	  }
	}
      }
    }

    void add_CGproduct(const SO3vecArray& x, const SO3vec& y, const int maxL=-1){
      assert(tau==GElib::CGproduct(x.tau,y.tau,maxL));

      int L1=x.getL(); 
      int L2=y.getL();
      vector<int> offs(tau.size(),0);
	
      for(int l1=0; l1<=L1; l1++){
	if(x.tau[l1]==0) continue;
	for(int l2=0; l2<=L2; l2++){
	  if(y.tau[l2]==0) continue;
	  for(int l=std::abs(l2-l1); l<=l1+l2 && (maxL<0 || l<=maxL); l++){
	    //array[l]->add_CGproduct(*x.array[l1],*y.parts[l2],offs[l]);
	    offs[l]+=(x.array[l1]->getn())*(y.parts[l2]->getn());
	  }
	}
      }
    }



  public: // ---- Spherical harmonics -----------------------------------------------------------------------

    /*
    static SO3vecArray  spharm(const int l, const int n, const Gtensor<float>& _x, const int _nbu=-1, const device& dev=0){
      assert(_x.k==1);
      assert(_x.dims[0]==3);
      float x=_x(0);
      float y=_x(1);
      float z=_x(2);
      //return SO3vecArray(engine::new_SO3vecArray_spharm(l,x(0),x(1),x(2),_nbu,dev.id()),l,1,_nbu);

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

      return SO3vecArray();
      //return newSO3vecArray(engine::new_SO3vecArray(R),l,1);
      //return SO3vecArray(Cengine_engine->push<new_SO3vecArray_from_Gtensor_op>(R,dev.id()),l,n);
    }
    */
    
    /*
    SO3vecArray static spharm(const int l, const int n, const float x, const float y, const float z, const int _nbu=-1, const device_id& dev=0){
      //return SO3vecArray(engine::new_SO3vecArray_spharm(l,x,y,z,_nbu,dev.id()),l,1,_nbu);
      return SO3vecArray(Cengine_engine->push<new_spharm_op>(l,n,x,y,z,_nbu,dev.id()),l,n);
    }
    */
 



  public: // ---- Not in-place operations --------------------------------------------------------------------

    
    //SO3vecArray plus(const SO3vecArray& x){
    //return SO3vecArray(Cengine_engine->push<ctensor_add_op>(hdl,x.hdl,dims),l,n,nbu);
    //}

    //SO3vecArray normalize() const{
      //return SO3vecArray(::Cengine::engine::ctensor_normalize_cols(hdl),l,n,nbu);
    //}

    //void add_normalize_back(const SO3vecArray& g, const SO3vecArray& x){
      //replace(hdl,::Cengine::engine::ctensor_add_normalize_cols_back(hdl,g.hdl,x.hdl));
    //}

    //SO3vecArray chunk(const int i, const int n=1) const {
    //return GELIB_SO3PARTARRAY_IMPL::chunk(1,i,n);
    //}

    /*
    SO3vecArray rotate(const SO3element& r){
      GELIB_SO3PARTARRAY_IMPL D(WignerMatrix<float>(l,r),dev);
      SO3vecArray R(l,n,cnine::fill::zero,dev);
      R.add_prod(D,*this);
      return R;
    }
    */

    /*
    GELIB_SO3PARTARRAY_IMPL fragment_norms() const{
      return column_norms();
    }

    SO3vecArray divide_fragments(const GELIB_SO3PARTARRAY_IMPL& N) const{
      return divide_columns(N);
    }
    */


  public: // ---- In-place operators -------------------------------------------------------------------------


    SO3vecArray& operator+=(const SO3vecArray& y){
      add(y);
      return *this;
    }

    SO3vecArray& operator-=(const SO3vecArray& y){
      subtract(y);
      return *this;
    }

 
  public: // ---- Binary operators ---------------------------------------------------------------------------


    SO3vecArray operator+(const SO3vecArray& y) const{
      SO3vecArray R(*this);
      R.add(y);
      return R;
    }

    SO3vecArray operator-( const SO3vecArray& y) const{
      SO3vecArray R(*this);
      R.subtract(y);
      return R;
    }

    SO3vecArray operator*(const cscalar& c) const{
      SO3vecArray R(*this,cnine::fill::zero);
      R.add(*this,c);
      return R;
    }

    /*
      SO3vecArray operator*(const ctensor& M) const{
      SO3vecArray R(adims,l,M.get_dims()[0],nbu,cnine::fill::zero);
      R.add_Mprod_AT<0>(*this,M);
      return R;
    }
    
    ctensor operator*(const Transpose<SO3vecArray>& _y) const{
      const SO3vecArray& y=_y.obj;
      assert(y.getl()==getl());
      ctensor_arr R(adims,Gdims(getn(),y.getn()),cnine::fill::zero);
      R.add_mprod_TA(y,*this);
      return R;
    }
    */
    

  public: // ---- I/O --------------------------------------------------------------------------------------


    string classname() const{
      return "GEnet::SO3vecArray";
    }

    string describe() const{
      return "SO3vecArray(l="+tau.str()+")";
    }

    //string str(const string indent="") const{
    //return gtensor().str_transp(indent);
    //}

    friend ostream& operator<<(ostream& stream, const SO3vecArray& x){
      stream<<x.str(); return stream;}

    };


  // ---- Post-class functions -------------------------------------------------------------------------------


  SO3vecArray operator*(const cnine::CscalarObj& c, const SO3vecArray& x){
    return x*c;
  }

  SO3vecArray CGproduct(const SO3vecArray& x, const SO3vecArray& y, const int maxL=-1){
    assert(x.adims==y.adims);
    SO3vecArray R(x.adims,CGproduct(x.tau,y.tau,maxL),x.get_nbu(),cnine::fill::zero,x.get_dev());
    R.add_CGproduct<0>(x,y,maxL);
    return R;
  }

  SO3vecArray CGproduct(const cnine::Outer<SO3vecArray,SO3vecArray>& args, const int maxL=-1){
    const SO3vecArray& x=args.obj0;
    const SO3vecArray& y=args.obj1;
    SO3vecArray R(cnine::Gdims(x.adims,y.adims),CGproduct(x.tau,y.tau,maxL),cnine::fill::zero,x.get_dev());
    R.add_CGproduct<1>(x,y,maxL);
    return R;
  }

  SO3vecArray CGproduct(const cnine::Convolve<SO3vecArray,SO3vecArray>& args, const int maxL=-1){
    const SO3vecArray& x=args.obj0;
    const SO3vecArray& y=args.obj1;
    SO3vecArray R(x.adims.convolve(y.adims),CGproduct(x.tau,y.tau,maxL),cnine::fill::zero,x.get_dev());
    R.add_CGproduct<2>(x,y,maxL);
    return R;
  }

  SO3vecArray CGproduct(const SO3vec& x, const SO3vecArray& y, const int maxL=-1){
    SO3vecArray R(y.adims,CGproduct(x.tau,y.tau,maxL),cnine::fill::zero,x.dev);
    R.add_CGproduct(x,y,maxL);
    return R;
  }

  SO3vecArray CGproduct(const SO3vecArray& x, const SO3vec& y, const int maxL=-1){
    SO3vecArray R(x.adims,CGproduct(x.tau,y.tau,maxL),cnine::fill::zero,y.dev);
    R.add_CGproduct(x,y,maxL);
    return R;
  }



}

#endif 

    /*
    SO3vecArray(const Gdims& _adims, const int _l, const int _n, const int _nbu=-1, const int _dev=0):
      GELIB_SO3PARTARRAY_IMPL(_adims,{2*_l+1,_n},_nbu,cnine::fill_raw(),_dev), l(_l), n(_n){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecArray(const Gdims& _adims, const int _l, const int _n, const FILLTYPE& fill, const int _dev=0):
      GELIB_SO3PARTARRAY_IMPL({2*_l+1,_n},fill,_dev), l(_l), n(_n){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecArray(const Gdims& _adims, const int _l, const int _n, const int _nbu, const FILLTYPE& fill, const int _dev=0):
      GELIB_SO3PARTARRAY_IMPL({2*_l+1,_n},_nbu,fill,_dev), l(_l), n(_n){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecArray(const Gdims& _adims, const int _l, const int _n, const FILLTYPE& fill, const device& _dev):
      SO3vecArray(_l,_n,-1,fill,_dev.id()){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    SO3vecArray(const Gdims& _adims, const int _l, const int _n, const int _nbu, const FILLTYPE& fill, 
      const device& _dev):
      SO3vecArray(_l,_n,_nbu,fill,_dev.id()){}
    */

