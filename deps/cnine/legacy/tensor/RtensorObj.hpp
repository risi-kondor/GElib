/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineRtensorObj
#define _CnineRtensorObj

#include "Cnine_base.hpp"
#include "RtensorA.hpp"
//#include "Dobject.hpp"
#include "ExprTemplates.hpp"
#include "RscalarObj.hpp"
#include "RtensorObj_helpers.hpp"
//#include "GenericOperators.hpp"


namespace cnine{


  class RtensorObj: public CNINE_RTENSOR_IMPL{
  public:

    using CNINE_RTENSOR_IMPL::CNINE_RTENSOR_IMPL; 

    static float dummy_scalar() {return 0;}

#ifdef WITH_FAKE_GRAD
    RtensorObj* grad=nullptr;
#endif 

    ~RtensorObj(){
#ifdef WITH_FAKE_GRAD
      if(!is_view) delete grad;
#endif 
    }


  public: // ---- Public Constructors ------------------------------------------------------------------------


    //RtensorObj(const Gdims& _dims, const int _dev=0): // this caused ambiguous overload
      //CNINE_RTENSOR_IMPL(_dims,_dev){}

    //RtensorObj(const Gdims& _dims, const int _nbu, const int _dev): // this caused ambiguous overload 
    //CNINE_RTENSOR_IMPL(_dims,_nbu,_dev){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    RtensorObj(const Gdims& _dims, const FILLTYPE& fill, const int _dev=0):
      CNINE_RTENSOR_IMPL(_dims,fill,_dev){}

    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //RtensorObj(const Gdims& _dims, const int _nbu, const FILLTYPE& fill, const int _dev=0):
    //CNINE_RTENSOR_IMPL(_dims,_nbu,fill,_dev){}
    
    RtensorObj(const Gdims& _dims, std::function<float(const int i, const int j)> fn):
      CNINE_RTENSOR_IMPL(_dims,fn){}


  public: // ---- Named constructors -------------------------------------------------------------------------


    static RtensorObj raw(const Gdims& _dims, const int nbd=-1, const int _dev=0){
      return RtensorObj(_dims,nbd,fill::raw,_dev);}
    static RtensorObj raw(const Gdims& _dims, const int nbd, const struct device& _dev){
      return RtensorObj(_dims,nbd,fill::raw,_dev.id());}
    static RtensorObj raw(const Gdims& _dims, const struct device& _dev){
      return RtensorObj(_dims,-1,fill::raw,_dev.id());}

    static RtensorObj zero(const Gdims& _dims, const int nbd=-1, const int _dev=0){
      return RtensorObj(_dims,nbd,fill::zero,_dev);}
    static RtensorObj zero(const Gdims& _dims, const int nbd, const struct device& _dev){
      return RtensorObj(_dims,nbd,fill::zero,_dev.id());}
    static RtensorObj zero(const Gdims& _dims, const struct device& _dev){
      return RtensorObj(_dims,-1,fill::zero,_dev.id());}

    static RtensorObj ones(const Gdims& _dims, const int nbd=-1, const int _dev=0){
      return RtensorObj(_dims,nbd,fill::ones,_dev);}
    static RtensorObj ones(const Gdims& _dims, const int nbd, const struct device& _dev){
      return RtensorObj(_dims,nbd,fill::ones,_dev.id());}
    static RtensorObj ones(const Gdims& _dims, const struct device& _dev){
      return RtensorObj(_dims,-1,fill::ones,_dev.id());}

    static RtensorObj identity(const Gdims& _dims, const int nbd=-1, const int _dev=0){
      return RtensorObj(_dims,nbd,fill::identity,_dev);}
    static RtensorObj identity(const Gdims& _dims, const int nbd, const struct device& _dev){
      return RtensorObj(_dims,nbd,fill::identity,_dev.id());}
    static RtensorObj identity(const Gdims& _dims, const struct device& _dev){
      return RtensorObj(_dims,-1,fill::identity,_dev.id());}

    static RtensorObj sequential(const Gdims& _dims, const int nbd=-1, const int _dev=0){
      return RtensorObj(_dims,nbd,fill::sequential,_dev);}
    static RtensorObj sequential(const Gdims& _dims, const int nbd, const struct device& _dev){
      return RtensorObj(_dims,nbd,fill::sequential,_dev.id());}
    static RtensorObj sequential(const Gdims& _dims, const struct device& _dev){
      return RtensorObj(_dims,-1,fill::sequential,_dev.id());}

    static RtensorObj gaussian(const Gdims& _dims, const int nbd=-1, const int _dev=0){
      return RtensorObj(_dims,nbd,fill::gaussian,_dev);}
    static RtensorObj gaussian(const Gdims& _dims, const int nbd, const struct device& _dev){
      return RtensorObj(_dims,nbd,fill::gaussian,_dev.id());}
    static RtensorObj gaussian(const Gdims& _dims, const struct device& _dev){
      return RtensorObj(_dims,-1,fill::gaussian,_dev.id());}
    

  public: // ---- Spawning -----------------------------------------------------------------------------------


    static RtensorObj* new_zeros_like(const RtensorObj& x){
      return new RtensorObj(x.get_dims(),fill_zero(),x.get_dev());
    }

    
  public: // ---- Copying ------------------------------------------------------------------------------------


    RtensorObj(const CNINE_RTENSOR_IMPL& x):
      CNINE_RTENSOR_IMPL(x){};
      
    RtensorObj(CNINE_RTENSOR_IMPL&& x):
      CNINE_RTENSOR_IMPL(std::move(x)){};
    //CNINE_RTENSOR_IMPL(x,"dummy"){cout<<"mv"<<endl;};
      
    RtensorObj(const RtensorObj& x):
      CNINE_RTENSOR_IMPL(x){
      #ifdef WITH_FAKE_GRAD
      if(x.grad) grad=new RtensorObj(x);
      #endif
    };
      
    RtensorObj(const RtensorObj& x, const int _dev):
      CNINE_RTENSOR_IMPL(x,_dev){
      #ifdef WITH_FAKE_GRAD
      if(x.grad) grad=new RtensorObj(x);
      #endif
    };
      
    RtensorObj(const RtensorObj& x, const struct device& _dev):
      CNINE_RTENSOR_IMPL(x,_dev.id()){
      #ifdef WITH_FAKE_GRAD
      if(x.grad) grad=new RtensorObj(x);
      #endif
    };
      
    RtensorObj(const RtensorObj& x, const fill_zero& dummy):
      RtensorObj(x.dims,x.get_nbu(),x.dev){}
      
    RtensorObj(const RtensorObj& x, const fill_view& dummy):
      CNINE_RTENSOR_IMPL(x,view_flag()){}
      
    RtensorObj(RtensorObj&& x):
      CNINE_RTENSOR_IMPL(std::move(x)){
      #ifdef WITH_FAKE_GRAD
      grad=x.grad;
      x.grad=nullptr;
      #endif
    };

    RtensorObj& operator=(const RtensorObj& x){
      CNINE_RTENSOR_IMPL::operator=(x);
      #ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
      if(x.grad) grad=new RtensorObj(x);
      #endif
      return *this;
    }

    RtensorObj& operator=(RtensorObj&& x){
      CNINE_RTENSOR_IMPL::operator=(std::move(x));
     #ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
      grad=x.grad;
      x.grad=nullptr;
      #endif
      return *this;
    }
    
    /*
    Dobject* clone() const{
      return new RtensorObj(*this);
    }

    Dobject* spawn(const fill_zero& fill) const{
      return new RtensorObj(CNINE_RTENSOR_IMPL::spawn(fill));
    }

    Dobject* spawn(const fill_zero& fill, const int _dev) const{
      return new RtensorObj(CNINE_RTENSOR_IMPL::spawn(fill),_dev);
    }

    Dobject* spawn(const fill_gaussian& fill) const{
      return new RtensorObj(CNINE_RTENSOR_IMPL::spawn(fill));
    }
    */

    //RtensorObj(RtensorObj& x, const view_flag& flag):
    //RtensorObj(CNINE_RTENSOR_IMPL(x,flag)){}
      

  public: // ---- Views --------------------------------------------------------------------------------------


    RtensorObj view(){
      return CNINE_RTENSOR_IMPL::view();
    }


  public: // ---- Conversions --------------------------------------------------------------------------------

    
    RtensorObj(const Conjugate<RtensorObj>& x):
      RtensorObj(x.obj.conj()){}

    RtensorObj(const Transpose<RtensorObj>& x):
      RtensorObj(x.obj.transp()){}

    //RtensorObj(const Hermitian<RtensorObj>& x):
    //RtensorObj(x.obj.herm()){}

    //RtensorObj(const Transpose<Conjugate<RtensorObj> >& x):
    //RtensorObj(x.obj.obj.herm()){}

    //RtensorObj(const Conjugate<Transpose<RtensorObj> >& x):
    //RtensorObj(x.obj.obj.herm()){}

    RtensorObj(const Gtensor<float>& x, const struct device& _dev):
      CNINE_RTENSOR_IMPL(x,_dev.id()){}

    as_shape_tmp<RtensorObj> as_shape(const Gdims& dims) const{
      return as_shape_tmp<RtensorObj>(*this,dims);
    }


  public: // ---- ATEN ---------------------------------------------------------------------------------------

    
#ifdef _WITH_ATEN

    static RtensorObj view(at::Tensor& T){
      return CNINE_RTENSOR_IMPL::view(T);
    }

#endif


  public: // ---- Transport ----------------------------------------------------------------------------------
  

    RtensorObj to_device(const int _dev){
      return RtensorObj(CNINE_RTENSOR_IMPL::to_device(_dev));
    }
  
  
  public: // ---- Access -------------------------------------------------------------------------------------


    int get_nbu() const{ 
      return nbu;
    }

    int get_k() const{ 
      return dims.size();
    }

    Gdims get_dims() const{ 
      return dims;
    }

    int get_dim(const int i) const{
      return dims[i];
    }

    int get_dev() const{
      return dev;
    }

    int get_device() const{
      return dev;
    }


  public: // ---- Experimental -------------------------------------------------------------------------------


    #ifdef WITH_FAKE_GRAD
    void add_to_grad(const RtensorObj& x){
      if(grad) grad->add(x);
      else grad=new RtensorObj(x);
    }

    RtensorObj& get_grad(){
      if(!grad) grad=RtensorObj::new_zeros_like(*this);
      return *grad;
    }

    RtensorObj view_of_grad(){
      if(!grad) grad=new_zeros_like(*this);
      return grad->view();
    }
    #endif 


  public: // ---- Get/set elements ---------------------------------------------------------------------------
 

    RscalarObj get(const Gindex& ix) const{
      CNINE_CHECK_RANGE(ix.check_range(dims));
      return RscalarObj(CNINE_RTENSOR_IMPL::get(ix));
    }

    RscalarObj get(const int i0) const{
      return RscalarObj(CNINE_RTENSOR_IMPL::get(i0));
    }

    RscalarObj get(const int i0, const int i1) const{
      return RscalarObj(CNINE_RTENSOR_IMPL::get(i0,i1));
    }

    RscalarObj get(const int i0, const int i1, const int i2) const{
      return RscalarObj(CNINE_RTENSOR_IMPL::get(i0,i1,i2));
    }

    RtensorObj& set(const Gindex& ix, const RscalarObj& v){
      CNINE_RTENSOR_IMPL::set(ix,v);
      return *this;
    }
    
    RtensorObj& set(const int i0, const RscalarObj& v){
      CNINE_RTENSOR_IMPL::set(i0,v);
      return *this;
    }
    
    RtensorObj& set(const int i0, const int i1, const RscalarObj& v){
      CNINE_RTENSOR_IMPL::set(i0,i1,v);
      return *this;
    }
    
    RtensorObj& set(const int i0, const int i1, const int i2, const RscalarObj& v){
      CNINE_RTENSOR_IMPL::set(i0,i1,i2,v);
      return *this;
    }


  public: // ---- Get/set value ------------------------------------------------------------------------------


    /*
    float get_value(const Gindex& ix) const{
      return CNINE_RTENSOR_IMPL::get_value(ix);
    }

    complex<float> get_value(const int i0) const{
      return CNINE_RTENSOR_IMPL::get_value(i0);
    }

    complex<float> get_value(const int i0, const int i1) const{
      return CNINE_RTENSOR_IMPL::get_value(i0,i1);
    }

    complex<float> get_value(const int i0, const int i1, const int i2) const{
      return CNINE_RTENSOR_IMPL::get_value(i0,i1,i2);
    }
    */

    RtensorObj& set_value(const Gindex& ix, float v){
      CNINE_RTENSOR_IMPL::set_value(ix,v);
      return *this;
    }
    
    RtensorObj& set_value(const int i0, float v){
      CNINE_RTENSOR_IMPL::set_value(i0,v);
      return *this;
    }
    
    RtensorObj& set_value(const int i0, const int i1, float v){
      CNINE_RTENSOR_IMPL::set_value(i0,i1,v);
      return *this;
    }
    
    RtensorObj& set_value(const int i0, const int i1, const int i2, float v){
      CNINE_RTENSOR_IMPL::set_value(i0,i1,i2,v);
      return *this;
    }
    
    float value(const Gindex& ix) const{
      return get_value(ix);}
    float value(const int i0) const{
      return get_value(i0);}
    float value(const int i0, const int i1) const{
      return get_value(i0,i1);}
    float value(const int i0, const int i1, const int i2) const{
      return get_value(i0,i1,i2);}

    RtensorObj& set(const Gindex& ix, float v){
      return set_value(ix,v);}
    RtensorObj& set(const int i0, float v){
      return set_value(i0,v);}
    RtensorObj& set(const int i0, const int i1, float v){
      return set_value(i0,i1,v);}
    RtensorObj& set(const int i0, const int i1, const int i2, float v){
      return set_value(i0,i1,i2,v);}


  public: // ---- CtensorElement -----------------------------------------------------------------------------
    

    RtensorObj_element operator()(const Gindex& ix){
      return RtensorObj_element(*this, ix);
    }

    RtensorObj_element operator()(const int i0){
      return RtensorObj_element(*this, Gindex(i0));
    }

    RtensorObj_element operator()(const int i0, const int i1){
      return RtensorObj_element(*this, Gindex(i0,i1));
    }

    RtensorObj_element operator()(const int i0, const int i1, const int i2){
      return RtensorObj_element(*this, Gindex(i0,i1,i2));
    }

    ConstRtensorObj_element operator()(const Gindex& ix) const{
      return ConstRtensorObj_element(*this, ix);
    }

    ConstRtensorObj_element operator()(const int i0) const{
      return ConstRtensorObj_element(*this, Gindex(i0));
    }

    ConstRtensorObj_element operator()(const int i0, const int i1) const{
      return ConstRtensorObj_element(*this, Gindex(i0,i1));
    }

    ConstRtensorObj_element operator()(const int i0, const int i1, const int i2) const{
      return ConstRtensorObj_element(*this, Gindex(i0,i1,i2));
    }


  public: // -------------------------------------------------------------------------------------------------

    /*
    RtensorObj& add_to_element(const Gindex& ix, const RscalarObj& v){
      replace(hdl,Cengine_engine->push<ctensor_add_to_element_op>(hdl,v.hdl,ix));
      return *this;
    }
    
    void add_element_into(RscalarObj& r, const Gindex& ix){
      replace(r.hdl,Cengine_engine->push<ctensor_add_element_op>(r.hdl,hdl,ix));
    }
    */

    /*
    int combined(const int a, const int b) const{
      return asCtensorB(hdl->node->obj).combined(a,b);
    }
    */

    void flush() const{
    }


  public: // ---- In-place operations ------------------------------------------------------------------------


    void clear(){
      set_zero();
    }


  public: // ---- Not in-place operations --------------------------------------------------------------------


    RtensorObj conj() const{
      return RtensorObj(CNINE_RTENSOR_IMPL::conj());
    }

    RtensorObj transp() const{
      return RtensorObj(CNINE_RTENSOR_IMPL::transp());
    }

    /*
    RtensorObj herm() const{
      return RtensorObj(CNINE_RTENSOR_IMPL::herm());
    }
    */

    RtensorObj plus(const RtensorObj& x) const{
      return RtensorObj(CNINE_RTENSOR_IMPL::plus(x));
    }

    RtensorObj apply(std::function<float(const float)> fn) const{
      return CNINE_RTENSOR_IMPL(*this,fn);
    }

    RtensorObj apply(std::function<float(const int i, const int j, const float)> fn) const{
      return CNINE_RTENSOR_IMPL(*this,fn);
    }


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add_mprod(const RtensorObj& x, const RtensorObj& y){
      add_Mprod_AA(x,y);
    }

    void add_mprod_AT(const RtensorObj& x, const RtensorObj& y){
      add_Mprod_AT(x,y);
    }

    void add_mprod_TA(const RtensorObj& x, const RtensorObj& y){
      add_Mprod_TA(x,y);
    }

    void add_mprod_AC(const RtensorObj& x, const RtensorObj& y){
      add_Mprod_AA(x,y);
    }

    void add_mprod_TC(const RtensorObj& x, const RtensorObj& y){
      add_Mprod_TA(x,y);
    }

    void add_mprod_AH(const RtensorObj& x, const RtensorObj& y){
      add_Mprod_AT(x,y);
    }

    void add_mprod_HA(const RtensorObj& x, const RtensorObj& y){
      add_Mprod_TA(x,y);
    }



  public: // ---- Slices and chunks --------------------------------------------------------------------------


    RtensorObj chunk(const int ix, const int offs, const int n){
      return CNINE_RTENSOR_IMPL::chunk(ix,offs,n);
    }

    RtensorObj slice(const int ix, const int offs){
      return CNINE_RTENSOR_IMPL::slice(ix,offs);
    }

    /*
    void add_to_slice(const int ix, const int offs, const RtensorObj& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_to_chunk(const int ix, const int offs, const RtensorObj& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_chunk_op>(hdl,x.hdl,ix,offs));
    }
    */

    void add_to_slices(const int ix, vector<const RtensorObj*> _v){
      vector<const CNINE_RTENSOR_IMPL*> v(_v.size());
      for(int i=0; i<_v.size(); i++) v[i]=_v[i];
      CNINE_RTENSOR_IMPL::add_to_slices(ix,v);
    }
    
    template<typename... Args>
    void add_to_slices(const int ix, const RtensorObj& x0, Args... args){
      add_to_slices(ix,const_variadic_unroller(x0,args...));
    }

    /*
    void add_slice(const RtensorObj& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_add_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void set_chunk(const RtensorObj& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_set_chunk_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_chunk(const RtensorObj& x, const int ix, const int offs, const int n){
      replace(hdl,Cengine_engine->push<ctensor_add_chunk_op>(hdl,x.hdl,ix,offs,n));
    }
    
    RtensorObj slice(const int ix, const int offs) const{
      RtensorObj R(dims.remove(ix),fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_slice_op>(R.hdl,hdl,ix,offs));
      return R;
    }

    RtensorObj chunk(const int ix, const int offs, const int n=1) const{
      Gdims Rdims=dims;
      Rdims[ix]=n;
      RtensorObj R(Rdims,fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_chunk_op>(R.hdl,hdl,ix,offs,n));
      return R;
    }
    */


  public: // ---- Into operations ----------------------------------------------------------------------------


    //void inp_into(RscalarObj& R, const RtensorObj& y) const{
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,y.hdl,dims));
    //}

    //void norm2_into(RscalarObj& R) const{
    //R.val=
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,hdl,dims));
    //}

    void add_norm2_back(const RscalarObj& g, const RtensorObj& x){
      add(x,g);
      add(x,g);
    }


  // ---- In-place operators ---------------------------------------------------------------------------------


    RtensorObj& operator+=(const RtensorObj& y){
      add(y);
      return *this;
    }

    RtensorObj& operator-=(const RtensorObj& y){
      subtract(y);
      return *this;
    }


  // ---- Binary operators -----------------------------------------------------------------------------------


    RtensorObj operator+(const RtensorObj& y) const{
      RtensorObj R(*this);
      R.add(y);
      return R;
    }

    RtensorObj operator-(const RtensorObj& y) const{
      RtensorObj R(*this);
      R.subtract(y);
      return R;
    }

    //RtensorObj operator*(const RscalarObj& c) const{
    //RtensorObj R(get_dims(),get_nbu(),fill::zero);
    //R.add(*this,c);
    //return R;
    //}

    /*
    RtensorObj operator*(const RtensorObj& y) const{
      int I=dims.combined(0,dims.k()-1);
      int J=y.dims.combined(1,y.dims.k());
      RtensorObj R({I,J},fill::zero);
      R.add_mprod(*this,y);
      return R;
    }
    */

    RtensorObj operator*(const Transpose<RtensorObj>& y) const{
      int I=dims.combined(0,dims.size()-1);
      int J=y.obj.dims.combined(0,y.obj.dims.size()-1);
      RtensorObj R(cnine::dims(I,J),fill::zero);
      R.add_mprod_AT(*this,y.obj);
      return R;
    }


  public: // ---- Normalization ------------------------------------------------------------------------------


    RtensorObj col_norms() const{
      Gdims _dims=get_dims();
      RtensorObj R(_dims.remove(_dims.size()-2),get_nbu(),fill::zero,dev);
      R.add_col_norms(*this);
      return R;
    }

    RtensorObj divide_cols(const RtensorObj& N) const{
      return RtensorObj(CNINE_RTENSOR_IMPL::divide_cols(N)); 
    }
    
    RtensorObj normalize_cols() const{
      return RtensorObj(CNINE_RTENSOR_IMPL::normalize_cols()); 
    }
    


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "GEnet::RtensorObj";
    }

    string describe() const{
      return "Ctensor"+dims.str();
    } 

    string str(string indent="") const{
      return CNINE_RTENSOR_IMPL::str(indent);
    }

    string str(const string indent, const float eps) const{
      return CNINE_RTENSOR_IMPL::str(indent);
    }

    string repr() const{
      return "<cnine::rtensor"+dims.str()+">";
    }

    friend ostream& operator<<(ostream& stream, const RtensorObj& x){
      stream<<x.str(); return stream;}

    //friend ostream& operator<<(ostream& stream, RtensorObj x){
    //stream<<x.str(); return stream;}

  };


  // ---- Post-class functions -------------------------------------------------------------------------------

  /*
  inline RtensorObj CtensorSeed::spawn(const fill_zero& fill){
    //if(nch<0) return new SO3partB(l,n,fill::zero,dev);
    return RtensorObj(dims,nbu,fill::zero,dev);
  }
  */

  inline RtensorObj_element::operator RscalarObj() const{
    return obj.get(ix);
  }

  inline RtensorObj_element& RtensorObj_element::operator=(const RscalarObj& x){
    obj.set(ix,x);    
    return *this;
  }

  inline float RtensorObj_element::get_value() const{
    return obj.get_value(ix);
  }
  
  inline RtensorObj_element& RtensorObj_element::set_value(const float x){
    obj.set_value(ix,x);
    return *this;
  }


  inline ConstRtensorObj_element::operator RscalarObj() const{
    return obj.get(ix);
  }

  inline ConstRtensorObj_element::operator float() const{
    return obj.get_value(ix);
  }

  inline float ConstRtensorObj_element::value() const{
    return obj.get_value(ix);
  }


  // ---------------------------------------------------------------------------------------------------------


}


#endif

