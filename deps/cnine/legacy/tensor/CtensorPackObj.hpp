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


#ifndef _CnineCtensorPackObj
#define _CnineCtensorPackObj

#include "Cnine_base.hpp"
#include "CtensorA.hpp"
//#include "Dobject.hpp"
#include "ExprTemplates.hpp"
#include "CscalarObj.hpp"
#include "CtensorObj.hpp"
#include "GdimsPack.hpp"
//#include "CtensorPackObj_helpers.hpp"
//#include "GenericOperators.hpp"


namespace cnine{


  class CtensorPackObj{ //: public CtensorObj{
  public:

    //typedef cnine::device device;
    typedef cnine::fill_pattern fill_pattern;
    typedef cnine::CoutLock CoutLock;

    typedef cnine::RscalarObj rscalar;
    typedef cnine::CscalarObj cscalar;
    typedef cnine::CtensorObj ctensor;

    int fmt=0;
    int dev=0; // is this redundant? 
    vector<CtensorObj*> tensors;
    bool is_view=false;

    #ifdef WITH_FAKE_GRAD
    CtensorPackObj* grad=nullptr;
    #endif 

    

    CtensorPackObj(){}

    ~CtensorPackObj(){
      for(auto p: tensors) delete p;  
      #ifdef WITH_FAKE_GRAD
      if(!is_view) delete grad;
      #endif 
    }


    // ---- Constructors --------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const FILLTYPE fill, const int _fmt, const int _dev): 
      dev(_dev), fmt(_fmt){
      const int n=_dims.size(); 
      if(fmt==0){
	for(int i=0; i<n; i++)
	  tensors.push_back(new CtensorObj(_dims[i],fill,_dev));
      }
      if(fmt==1){
	CNINE_UNIMPL();
      }
    }


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const int n, const Gdims& _dims, const FILLTYPE fill, const int _fmt=0, const int _dev=0): 
      dev(_dev), fmt(_fmt){
      if(fmt==0){
	for(int i=0; i<n; i++){
	  tensors.push_back(new CtensorObj(_dims,fill,_dev));
	}
      }
      if(fmt==1){
	CNINE_UNIMPL();
      }
    }
    

    // ---- Constructors ------------------------------------------------------------------------------------


    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const FILLTYPE fill): 
      CtensorPackObj(_dims,fill,0,0){}

    template<typename FILLTYPE, typename = typename 
	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const FILLTYPE fill, const pack_format& _format): 
      CtensorPackObj(_dims,fill,toint(_format),0){}

    //template<typename FILLTYPE, typename = typename 
    //	     std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //CtensorPackObj(const GdimsPack& _dims, const FILLTYPE fill, const device& _device): 
    //CtensorPackObj(_dims,fill,0,_device.id()){}

    template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    CtensorPackObj(const GdimsPack& _dims, const FILLTYPE fill, const pack_format& _format, const struct device& _device): 
      CtensorPackObj(_dims,fill,toint(_format),_device.id()){}
  
  
  public: // ---- Static constructors --------------------------------------------------------------------------


    static CtensorPackObj raw(const int _n, const Gdims& _dims, const int _dev=0){
      return CtensorPackObj(_n,_dims,fill_raw(),0,_dev);}

    static CtensorPackObj zero(const int _n, const Gdims& _dims, const int _dev=0){
      return CtensorPackObj(_n,_dims,cnine::fill::zero,0,_dev);}
    //static CtensorPackObj zero(const int _n, const Gdims& _dims, const pack_format& _format){
    //return CtensorPackObj(_n,_dims,-1,cnine::fill::zero,toint(_format));}
    //static CtensorPackObj zero(const int _n, const Gdims& _dims, const device& _device){
    //return CtensorPackObj(_n,_dims,-1,cnine::fill::zero,_device.id());}
    //static CtensorPackObj zero(const int _n, const Gdims& _dims, const pack_format& _format, const device& _device){
    //return CtensorPackObj(_n,_dims,-1,cnine::fill::zero,toint(_format),_device.id());}
    
    static CtensorPackObj zero(const GdimsPack& _dims, const int _dev=0){
      return CtensorPackObj(_dims,cnine::fill::zero,0,_dev);}
    //static CtensorPackObj zero(const GdimsPack& _dims, const pack_format& _format){
    //  return CtensorPackObj(_dims,-1,cnine::fill::zero,toint(_format));}
    //static CtensorPackObj zero(const GdimsPack& _dims, const device& _device){
    //return CtensorPackObj(_dims,-1,cnine::fill::zero,_device.id());}
    //static CtensorPackObj zero(const GdimsPack& _dims, const pack_format& _format, const device& _device){
    //return CtensorPackObj(_dims,-1,cnine::fill::zero,toint(_format),_device.id());}
    
    static CtensorPackObj ones(const int _n, const Gdims& _dims, const int _dev=0){
      return CtensorPackObj(_n,_dims,fill_ones(),0,_dev);}

    static CtensorPackObj ones(const GdimsPack& _dims, const int _dev=0){
      return CtensorPackObj(_dims,cnine::fill::ones,0,_dev);}
    //static CtensorPackObj ones(const GdimsPack& _dims, const pack_format& _format){
    //return CtensorPackObj(_dims,-1,cnine::fill::ones,_format);}
    //static CtensorPackObj ones(const GdimsPack& _dims, const device& _device){
    //return CtensorPackObj(_dims,-1,cnine::fill::ones,_device);}
    //static CtensorPackObj ones(const GdimsPack& _dims, const pack_format& _format, const device& _device){
    //return CtensorPackObj(_dims,-1,cnine::fill::ones,_format,_device);}
    
    static CtensorPackObj sequential(const int _n, const Gdims& _dims, const int _dev=0){
      return CtensorPackObj(_n,_dims,cnine::fill::sequential,0,_dev);}
    //static CtensorPackObj sequential(const int _n, const Gdims& _dims, const pack_format& _format){
    //return CtensorPackObj(_n,_dims,-1,cnine::fill::sequential,toint(_format));}
    //static CtensorPackObj sequential(const int _n, const Gdims& _dims, const device& _device){
    //return CtensorPackObj(_n,_dims,-1,cnine::fill::sequential,0,_device.id());}
    //static CtensorPackObj sequential(const int _n, const Gdims& _dims, const pack_format& _format, const device& _device){
    //return CtensorPackObj(_n,_dims,-1,cnine::fill::sequential,toint(_format),_device.id());}
    
    static CtensorPackObj sequential(const GdimsPack& _dims, const int _dev=0){
      return CtensorPackObj(_dims,cnine::fill::sequential,0,_dev);}
    //static CtensorPackObj sequential(const GdimsPack& _dims, const pack_format& _format){
    //return CtensorPackObj(_dims,-1,cnine::fill::sequential,_format);}
    //static CtensorPackObj sequential(const GdimsPack& _dims, const device& _device){
    //return CtensorPackObj(_dims,-1,cnine::fill::sequential,_device);}
    //static CtensorPackObj sequential(const GdimsPack& _dims, const pack_format& _format, const device& _device){
    //return CtensorPackObj(_dims,-1,cnine::fill::sequential,_format,_device);}
    
    static CtensorPackObj gaussian(const int _n, const Gdims& _dims, const int _dev=0){
      return CtensorPackObj(_n,_dims,cnine::fill::gaussian,0,_dev);}
    //static CtensorPackObj gaussian(const int _n, const Gdims& _dims, const pack_format& _format){
    //return CtensorPackObj(_n,_dims,-1,cnine::fill::gaussian,toint(_format));}
    //static CtensorPackObj gaussian(const int _n, const Gdims& _dims, const device& _device){
    //return CtensorPackObj(_n,_dims,-1,cnine::fill::gaussian,_device.id());}
    //static CtensorPackObj gaussian(const int _n, const Gdims& _dims, const pack_format& _format, const device& _device){
    //return CtensorPackObj(_n,_dims,-1,cnine::fill::gaussian,toint(_format),_device.id());}
    
    static CtensorPackObj gaussian(const GdimsPack& _dims, const int _dev=0){
      return CtensorPackObj(_dims,cnine::fill::gaussian,0,_dev);}
    //static CtensorPackObj gaussian(const GdimsPack& _dims, const pack_format& _format){
    //return CtensorPackObj(_dims,-1,cnine::fill::gaussian,toint(_format));}
    //static CtensorPackObj gaussian(const GdimsPack& _dims, const device& _device){
    //return CtensorPackObj(_dims,-1,cnine::fill::gaussian,_device.id());}
    //static CtensorPackObj gaussian(const GdimsPack& _dims, const pack_format& _format, const device& _device){
    //return CtensorPackObj(_dims,-1,cnine::fill::gaussian,toint(_format),_device.id());}

    
    static CtensorPackObj zeros_like(const CtensorPackObj& x){
      CtensorPackObj R;
      for(auto p: x.tensors)
	if(p) R.tensors.push_back(new CtensorObj(*p,fill_zero()));
	else R.tensors.push_back(nullptr);
      return R;
    }


  public: // ---- Spawning -----------------------------------------------------------------------------------


    static CtensorPackObj* new_zeros_like(const CtensorPackObj& x){
      return new CtensorPackObj(x,fill_zero());
    }

      
  public: // ---- Copying ------------------------------------------------------------------------------------


    CtensorPackObj(const CtensorPackObj& x):
      fmt(x.fmt), dev(x.dev){
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(*p));
	else tensors.push_back(nullptr);
    };
      
    CtensorPackObj(const CtensorPackObj& x, const int _dev):
      fmt(x.fmt), dev(_dev){
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(*p,_dev));
	else tensors.push_back(nullptr);
      #ifdef WITH_FAKE_GRAD
      if(x.grad) grad=new CtensorPackObj(x);
      #endif
    };
      
    //CtensorPackObj(const CtensorPackObj& x, const device& _dev):
    //CtensorObj(x,_dev.id()), nbu(x.nbu), fmt(x.fmt), dev(x.dev){
    //for(auto p: x.tensors)
    //if(p) tensors.push_back(new CtensorObj(*p,_dev.id()));
    //else tensors.push_back(nullptr);
    //};
      
    CtensorPackObj(const CtensorPackObj& x, const fill_zero& dummy):
      fmt(x.fmt), dev(x.dev){
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(*p,dummy));
	else tensors.push_back(nullptr);
    };
      
    CtensorPackObj(CtensorPackObj&& x):
      fmt(x.fmt), dev(x.dev){
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(std::move(*p)));
	else tensors.push_back(nullptr);
      #ifdef WITH_FAKE_GRAD
      grad=x.grad;
      x.grad=nullptr;
      #endif
    };

    CtensorPackObj& operator=(const CtensorPackObj& x){
      for(auto p: tensors) delete p;
      tensors.clear(); 
      for(auto p: x.tensors)
	if(p) tensors.push_back(new CtensorObj(*p));
	else tensors.push_back(nullptr);
      #ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
      if(x.grad) grad=new CtensorPackObj(x);
      #endif
      return *this;
    }

    CtensorPackObj& operator=(CtensorPackObj&& x){
      for(auto p: tensors) delete p;
      tensors.clear(); 
      for(auto p: x.tensors)
	tensors.push_back(p);
      #ifdef WITH_FAKE_GRAD
      if(grad) delete grad;
      grad=x.grad;
      x.grad=nullptr;
      #endif
      return *this;
    }

    
  public: // ---- Views -------------------------------------------------------------------------------


    CtensorPackObj view(){
      CtensorPackObj R;
      foreach_tensor([&](CtensorObj& x){R.tensors.push_back(new CtensorObj(x.view()));});
      R.is_view=true;
      return R;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    /*
    CtensorPackObj(const Conjugate<CtensorPackObj>& x):
      CtensorPackObj(x.obj.conj()){}

    CtensorPackObj(const Transpose<CtensorPackObj>& x):
      CtensorPackObj(x.obj.transp()){}

    CtensorPackObj(const Hermitian<CtensorPackObj>& x):
      CtensorPackObj(x.obj.herm()){}

    CtensorPackObj(const Transpose<Conjugate<CtensorPackObj> >& x):
      CtensorPackObj(x.obj.obj.herm()){}

    CtensorPackObj(const Conjugate<Transpose<CtensorPackObj> >& x):
      CtensorPackObj(x.obj.obj.herm()){}

    CtensorPackObj(const Gtensor<complex<float> >& x, const device& _dev=device(0)):
      CNINE_CTENSOR_IMPL(x,_dev.id()){}
    */

  public: // ---- ATen --------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    CtensorPackObj(vector<at::Tensor>& v){
      for(auto& p: v)
	tensors.push_back(new CtensorObj(p));
    }

    static CtensorPackObj view(vector<at::Tensor>& v){
      CtensorPackObj R;
      for(auto p:v)
	R.tensors.push_back(CtensorObj::viewp(p));
      return R;
    }

    vector<at::Tensor> torch(){
      vector<at::Tensor> R;
      for(auto p: tensors)
	R.push_back(p->torch());
      return R;
    }

#endif

 
  public: // ---- Transport ----------------------------------------------------------------------------------


    CtensorPackObj& move_to_device(const int _dev){
      foreach_tensor([&](CtensorObj& x){x.move_to_device(_dev);});
      dev=_dev;
      return *this;
    }
    
    CtensorPackObj to_device(const int _dev) const{
      CtensorPackObj R;
      foreach_tensor([&](const CtensorObj& x){
	  R.tensors.push_back(new CtensorObj(x.to_device(_dev)));});
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    Gdims get_dims(const int i) const{
      assert(i<tensors.size());
      return tensors[i]->get_dims();
    }

    int get_dev() const{
      if(tensors.size()==0) return 0;
      return tensors[0]->get_dev();
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void foreach_tensor(const std::function<void(CtensorObj& x)>& lambda){
      for(int i=0; i<tensors.size(); i++)
	lambda(*tensors[i]);
    }

    void foreach_tensor(const std::function<void(const CtensorObj& x)>& lambda) const{
      for(int i=0; i<tensors.size(); i++)
	lambda(*tensors[i]);
    }

    void foreach_tensor(const CtensorPackObj& ypack, 
      const std::function<void(const CtensorObj& x, const CtensorObj& y)>& lambda) const{
      CNINE_NTENS_SAME(ypack);
      for(int i=0; i<tensors.size(); i++)
	lambda(*tensors[i],*ypack.tensors[i]);
    }


  public: // ---- Experimental -------------------------------------------------------------------------------


    #ifdef WITH_FAKE_GRAD
    void add_to_grad(const CtensorPackObj& x){
      if(grad) grad->add(x);
      else grad=new CtensorPackObj(x);
    }

    CtensorPackObj& get_grad(){
      if(!grad) grad=new_zeros_like(*this);
      return *grad;
    }

    CtensorPackObj view_of_grad(){
      if(!grad) grad=new_zeros_like(*this);
      return grad->view();
    }
    #endif 


  public: // ---- Get/set elements ---------------------------------------------------------------------------
 


    /*
    CscalarObj get(const Gindex& ix) const{
      return CscalarObj(CNINE_CTENSOR_IMPL::get(ix));
    }

    CscalarObj get(const int i0) const{
      return CscalarObj(CNINE_CTENSOR_IMPL::get(i0));
    }

    CscalarObj get(const int i0, const int i1) const{
      return CscalarObj(CNINE_CTENSOR_IMPL::get(i0,i1));
    }

    CscalarObj get(const int i0, const int i1, const int i2) const{
      return CscalarObj(CNINE_CTENSOR_IMPL::get(i0,i1,i2));
    }

    CtensorPackObj& set(const Gindex& ix, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(ix,v);
      return *this;
    }
    
    CtensorPackObj& set(const int i0, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(i0,v);
      return *this;
    }
    
    CtensorPackObj& set(const int i0, const int i1, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(i0,i1,v);
      return *this;
    }
    
    CtensorPackObj& set(const int i0, const int i1, const int i2, const CscalarObj& v){
      CNINE_CTENSOR_IMPL::set(i0,i1,i2,v);
      return *this;
    }
    */


  public: // ---- Get/set value ------------------------------------------------------------------------------


    /*
    complex<float> get_value(const Gindex& ix) const{
      return CNINE_CTENSOR_IMPL::get_value(ix);
    }

    complex<float> get_value(const int i0) const{
      return CNINE_CTENSOR_IMPL::get_value(i0);
    }

    complex<float> get_value(const int i0, const int i1) const{
      return CNINE_CTENSOR_IMPL::get_value(i0,i1);
    }

    complex<float> get_value(const int i0, const int i1, const int i2) const{
      return CNINE_CTENSOR_IMPL::get_value(i0,i1,i2);
    }
    */

    /*
    CtensorPackObj& set_value(const Gindex& ix, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(ix,v);
      return *this;
    }
    
    CtensorPackObj& set_value(const int i0, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(i0,v);
      return *this;
    }
    
    CtensorPackObj& set_value(const int i0, const int i1, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(i0,i1,v);
      return *this;
    }
    
    CtensorPackObj& set_value(const int i0, const int i1, const int i2, complex<float> v){
      CNINE_CTENSOR_IMPL::set_value(i0,i1,i2,v);
      return *this;
    }
    
    complex<float> value(const Gindex& ix) const{
      return get_value(ix);}
    complex<float> value(const int i0) const{
      return get_value(i0);}
    complex<float> value(const int i0, const int i1) const{
      return get_value(i0,i1);}
    complex<float> value(const int i0, const int i1, const int i2) const{
      return get_value(i0,i1,i2);}

    CtensorPackObj& set(const Gindex& ix, complex<float> v){
      return set_value(ix,v);}
    CtensorPackObj& set(const int i0, complex<float> v){
      return set_value(i0,v);}
    CtensorPackObj& set(const int i0, const int i1, complex<float> v){
      return set_value(i0,i1,v);}
    CtensorPackObj& set(const int i0, const int i1, const int i2, complex<float> v){
      return set_value(i0,i1,i2,v);}
    */




  public: // ---- In-place operations ------------------------------------------------------------------------


    void clear(){
      //set_zero();
    }


  public: // ---- Not in-place operations --------------------------------------------------------------------

    /*
    CtensorPackObj conj() const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::conj());
    }

    CtensorPackObj transp() const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::transp());
    }

    CtensorPackObj herm() const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::herm());
    }

    CtensorPackObj plus(const CtensorPackObj& x){
      return CtensorPackObj(CNINE_CTENSOR_IMPL::plus(x));
    }

    CtensorPackObj apply(std::function<complex<float>(const complex<float>)> fn) const{
      return CNINE_CTENSOR_IMPL(*this,fn);
    }

    CtensorPackObj apply(std::function<complex<float>(const int i, const int j, const complex<float>)> fn) const{
      return CNINE_CTENSOR_IMPL(*this,fn);
    }
    */


  public: // ---- Cumulative operations ----------------------------------------------------------------------


    void add(const CtensorPackObj& x){
      CNINE_NTENS_SAME(x);
      if(fmt==0){
	for(int l=0; l<tensors.size(); l++)
	  if(tensors[l]) tensors[l]->add(*x.tensors[l]);
      }
      if(fmt==1)
	CNINE_UNIMPL();
    }

    void add(const CtensorPackObj& x, const complex<float> c){
      CNINE_NTENS_SAME(x);
      if(fmt==0){
	for(int l=0; l<tensors.size(); l++)
	  if(tensors[l]) tensors[l]->add(*x.tensors[l],c);
      }
      if(fmt==1)
	CNINE_UNIMPL();
    }

    void add(const CtensorPackObj& x, const rscalar& c){
      CNINE_NTENS_SAME(x);
      if(fmt==0){
	for(int l=0; l<tensors.size(); l++)
	  if(tensors[l]) tensors[l]->add(*x.tensors[l],c);
      }
      if(fmt==1)
	CNINE_UNIMPL();
    }

    void add(const CtensorPackObj& x, const cscalar& c){
      CNINE_NTENS_SAME(x);
      if(fmt==0){
	for(int l=0; l<tensors.size(); l++)
	  if(tensors[l]) tensors[l]->add(*x.tensors[l],c);
      }
      if(fmt==1)
	CNINE_UNIMPL();
    }

    void subtract(const CtensorPackObj& x){
      CNINE_NTENS_SAME(x);
      if(fmt==0){
	for(int l=0; l<tensors.size(); l++)
	  if(tensors[l]) tensors[l]->subtract(*x.tensors[l]);
      }
      if(fmt==1)
	CNINE_UNIMPL();
    }

    void add_mprod(const CtensorPackObj& x, const CtensorPackObj& y){
      CNINE_NTENS_SAME(x);
      CNINE_NTENS_SAME(y);
      for(int i=0; i<tensors.size(); i++)
	tensors[i]->add_mprod(*x.tensors[i],*y.tensors[i]);
    }

    /*
    void add_mprod_AT(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_AT<0>(x,y);
    }

    void add_mprod_TA(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_TA<0>(x,y);
    }

    void add_mprod_AC(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_AA<2>(x,y);
    }

    void add_mprod_TC(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_TA<2>(x,y);
    }

    void add_mprod_AH(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_AT<2>(x,y);
    }

    void add_mprod_HA(const CtensorPackObj& x, const CtensorPackObj& y){
      add_Mprod_TA<1>(x,y);
    }
    */


  public: // ---- Slices and chunks --------------------------------------------------------------------------

    /*
    CtensorPackObj chunk(const int ix, const int offs, const int n){
      return CNINE_CTENSOR_IMPL::chunk(ix,offs,n);
    }

    CtensorPackObj slice(const int ix, const int offs){
      return CNINE_CTENSOR_IMPL::slice(ix,offs);
    }
    */

    /*
    void add_to_slice(const int ix, const int offs, const CtensorPackObj& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_to_chunk(const int ix, const int offs, const CtensorPackObj& x){
      replace(hdl,Cengine_engine->push<ctensor_add_to_chunk_op>(hdl,x.hdl,ix,offs));
    }
    */

    /*
    void add_to_slices(const int ix, vector<const CtensorPackObj*> _v){
      vector<const CNINE_CTENSOR_IMPL*> v(_v.size());
      for(int i=0; i<_v.size(); i++) v[i]=_v[i];
      CNINE_CTENSOR_IMPL::add_to_slices(ix,v);
    }
    
    template<typename... Args>
    void add_to_slices(const int ix, const CtensorPackObj& x0, Args... args){
      add_to_slices(ix,const_variadic_unroller(x0,args...));
    }
    */

    /*
    void add_slice(const CtensorPackObj& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_add_slice_op>(hdl,x.hdl,ix,offs));
    }
    
    void set_chunk(const CtensorPackObj& x, const int ix, const int offs){
      replace(hdl,Cengine_engine->push<ctensor_set_chunk_op>(hdl,x.hdl,ix,offs));
    }
    
    void add_chunk(const CtensorPackObj& x, const int ix, const int offs, const int n){
      replace(hdl,Cengine_engine->push<ctensor_add_chunk_op>(hdl,x.hdl,ix,offs,n));
    }
    
    CtensorPackObj slice(const int ix, const int offs) const{
      CtensorPackObj R(dims.remove(ix),fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_slice_op>(R.hdl,hdl,ix,offs));
      return R;
    }

    CtensorPackObj chunk(const int ix, const int offs, const int n=1) const{
      Gdims Rdims=dims;
      Rdims[ix]=n;
      CtensorPackObj R(Rdims,fill::zero);
      replace(R.hdl,Cengine_engine->push<ctensor_add_chunk_op>(R.hdl,hdl,ix,offs,n));
      return R;
    }
    */


  public: // ---- Into operations ----------------------------------------------------------------------------


    //void inp_into(CscalarObj& R, const CtensorPackObj& y) const{
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,y.hdl,dims));
    //}

    //void norm2_into(CscalarObj& R) const{
    //R.val=
      //replace(R.hdl,Cengine_engine->push<ctensor_add_inp_op>(R.hdl,hdl,hdl,dims));
    //}

  /*
    void add_norm2_back(const CscalarObj& g, const CtensorPackObj& x){
      add(x,g);
      add_cconj(x,g);
    }
  */

public: // ---- In-place operators ---------------------------------------------------------------------------------


    CtensorPackObj& operator+=(const CtensorPackObj& y){
      add(y);
      return *this;
    }

    CtensorPackObj& operator-=(const CtensorPackObj& y){
      subtract(y);
      return *this;
    }


  // ---- Binary operators -----------------------------------------------------------------------------------


  CtensorPackObj operator+(const CtensorPackObj& y) const{
      CtensorPackObj R(*this);
      R.add(y);
      return R;
    }

    CtensorPackObj operator-(const CtensorPackObj& y) const{
      CtensorPackObj R(*this);
      R.subtract(y);
      return R;
    }

    CtensorPackObj operator*(const CscalarObj& c) const{
      CtensorPackObj R(*this,fill::zero);
      R.add(*this,c);
      return R;
    }

    
    CtensorPackObj operator*(const CtensorPackObj& y) const{
      CNINE_NTENS_SAME(y);
      CtensorPackObj R;
      for(int i=0; i<tensors.size(); i++){
	assert(tensors[i]->ndims()==2);
	assert(y.tensors[i]->ndims()==2);
	R.tensors.push_back(new CtensorObj(Gdims({tensors[i]->get_dim(0),y.tensors[i]->get_dim(1)}),fill_zero()));
      }
      R.add_mprod(*this,y);
      return R;
    }

  /*
    CtensorPackObj operator*(const Transpose<CtensorPackObj>& y) const{
      int I=dims.combined(0,dims.k()-1);
      int J=y.obj.dims.combined(0,y.obj.dims.k()-1);
      CtensorPackObj R({I,J},fill::zero);
      R.add_mprod_AT(*this,y.obj);
      return R;
    }
  */

  public: // ---- Scalar-valued operations -------------------------------------------------------------------


    complex<float> inp(const CtensorPackObj& y) const{
      CNINE_DEVICE_SAME(y);
      CNINE_NTENS_SAME(y);
      complex<float> r=0;
      foreach_tensor(y,[&](const CtensorObj& a, const CtensorObj& b){r+=a.inp(b);});
      return r;
    }

    float norm2() const{
      float r=0;
      foreach_tensor([&](const CtensorObj& a){r+=a.norm2();});
      return r;
    }


  public: // ---- Normalization ------------------------------------------------------------------------------

    /*
    CtensorPackObj col_norms() const{
      Gdims _dims=get_dims();
      CtensorPackObj R(_dims.remove(_dims.size()-2),get_nbu(),fill::zero,dev);
      R.add_col_norms(*this);
      return R;
    }

    CtensorPackObj divide_cols(const CtensorPackObj& N) const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::divide_cols(N)); 
    }
    
    CtensorPackObj normalize_cols() const{
      return CtensorPackObj(CNINE_CTENSOR_IMPL::normalize_cols()); 
    }
    */


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Cnine::CtensorPackObj";
    }

    string describe() const{
      return "CtensorPack";
    } 

    string str(const string indent="") const{
      //if(fmt==1) return CNINE_CTENSOR_IMPL::str(indent);
      ostringstream oss;
      for(int l=0; l<tensors.size(); l++){
	if(!tensors[l]) continue;
	oss<<indent<<"tensor "<<l<<":\n";
	oss<<tensors[l]->str(indent+"  ");
	oss<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CtensorPackObj& x){
      stream<<x.str(); return stream;}

    //friend ostream& operator<<(ostream& stream, CtensorPackObj x){
    //stream<<x.str(); return stream;}

  };


  // ---- Post-class functions -------------------------------------------------------------------------------


  inline complex<float> inp(const CtensorPackObj& x, const CtensorPackObj& y){
    return x.inp(y);
  }

  inline float norm2(const CtensorPackObj& x){
    return x.norm2();
  }


  // ---------------------------------------------------------------------------------------------------------


}


#endif


