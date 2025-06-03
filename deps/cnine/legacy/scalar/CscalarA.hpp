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


#ifndef _CscalarA
#define _CscalarA

#include "Cnine_base.hpp"
#include "CnineObject.hpp"
#include "RscalarA.hpp"

extern default_random_engine rndGen;


namespace cnine{


  class CscalarA: public CnineObject{ //, public CnineBackendObject{
  public:

    int nbu=-1;

    complex<float> val;
    complex<float>* arr=nullptr;

    CscalarA(){
    }

    ~CscalarA(){
      delete[] arr;
    }

    string classname() const{
      return "CscalarA";
    }

    string describe() const{
      if(nbu>=0) return "CscalarA["+to_string(nbu)+"]";
      return "CscalarA";
    }


  private: // ---- Private Constructors ---------------------------------------------------------------------


    void reallocate(){
      delete[] arr;
      if(nbu==-1) return;
      arr=new complex<float>[nbu];
    }


  public: // ---- Filled constructors -----------------------------------------------------------------------


    CscalarA(const fill_raw& fill){}

    CscalarA(const fill_zero& fill): val(0){}
 
    CscalarA(const fill_ones& fill): val(1){}

    CscalarA(const complex<float> c): val(c){}
 
    CscalarA(const float c): val(c){}
 
    CscalarA(const fill_gaussian& fill){
      normal_distribution<float> distr;
      val=complex<float>(distr(rndGen),distr(rndGen));
    }

    CscalarA(const fill_gaussian& fill, const float c){
      normal_distribution<float> distr;
      val=complex<float>(c*distr(rndGen),c*distr(rndGen));
    }

    CscalarA(const int _nbu, const fill_raw& fill): 
      nbu(_nbu){
      reallocate();
    }

    CscalarA(const int _nbu, const fill_zero& fill): 
      CscalarA(_nbu,fill::raw){
      if(nbu==-1) val=0; 
      else std::fill(arr,arr+nbu,0);
    }
 
    CscalarA(const int _nbu, const fill_gaussian& fill):
      CscalarA(_nbu,fill::raw){
      normal_distribution<float> distr;
      if(nbu==-1) val=distr(rndGen);
      else for(int i=0; i<nbu; i++) 
	arr[i]=complex<float>(distr(rndGen),distr(rndGen));
    }

    CscalarA(const int _nbu, const fill_gaussian& fill, const float c):
      CscalarA(_nbu,fill::raw){
      normal_distribution<float> distr;
      if(nbu==-1) val=c*distr(rndGen);
      else 
	for(int i=0; i<nbu; i++) 
	  arr[i]=complex<float>(c*distr(rndGen),c*distr(rndGen));
    }

    CscalarA(const int _nbu, const complex<float> c): 
      CscalarA(_nbu,fill::raw){
      if(nbu==-1) val=c; 
      else std::fill(arr,arr+nbu,c);
    }
 
    CscalarA(const int _nbu, const float c): 
      CscalarA(_nbu,fill::raw){
      if(nbu==-1) val=c; 
      else std::fill(arr,arr+nbu,c);
    }
 
    CscalarA(const int _nbu, const double c): 
      CscalarA(_nbu,fill::raw){
      if(nbu==-1) val=c; 
      else std::fill(arr,arr+nbu,c);
    }
 
    CscalarA(const CscalarA& x, std::function<complex<float>(const complex<float>)> fn):
      CscalarA(x.nbu,fill::raw){
      if(nbu==-1) val=fn(x.val); 
      else for(int i=0; i<nbu; i++) arr[i]=fn(x.arr[i]);
    }



  public: // ---- Copying -----------------------------------------------------------------------------------


    CscalarA(const CscalarA& x): 
      nbu(x.nbu){
      if(nbu==-1) val=x.val;
      else{
	reallocate();
	std::copy(x.arr,x.arr+nbu,arr);
      }
    }

    CscalarA(CscalarA&& x): 
      nbu(x.nbu){
      if(nbu==-1) val=x.val;
      else{
	arr=x.arr;
	x.arr=nullptr;
      }
    }

    CscalarA& operator=(const CscalarA& x){
      delete[] arr;
      arr=nullptr;
      nbu=x.nbu;
      if(nbu==-1){
	val=x.val;
      }else{
	arr=new complex<float>[nbu];
	std::copy(x.arr,x.arr+nbu,arr);
      }
      return *this;
    }
    
    CscalarA& operator=(CscalarA&& x){
      delete[] arr;
      arr=nullptr;
      nbu=x.nbu;
      if(nbu==-1){
	val=x.val;
      }else{
	arr=x.arr;
	x.arr=nullptr;
      }      
      return *this;
    }



  public: // ---- Conversions -------------------------------------------------------------------------------

    
    CscalarA(const vector<complex<float> >& v){
      if(v.size()==1){
	nbu=-1;
	val=v[0];
      }else{
	nbu=v.size();
	arr=new complex<float>[nbu];
	for(int i=0; i<nbu; i++) 
	  arr[i]=v[i];
      }
    }

    operator vector<complex<float> >(){
      if(nbu==-1){return vector<complex<float> >(1,val);}
      vector<complex<float> > R(nbu);
      for(int i=0; i<nbu; i++)
	R[i]=arr[i];
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getnbu() const{
      return nbu;
    }

    complex<float> get_value() const{
      assert(nbu==-1);
      return val;
    }

    complex<float> value() const{
      assert(nbu==-1);
      return val;
    }

    CscalarA& set_value(complex<float> x){
      assert(nbu==-1);
      val=x;
      return *this;
    }
    
    RscalarA real() const{
      RscalarA R(nbu,fill::raw);
      if(nbu==-1) R.val=val.real();
      else for(int i=0; i<nbu; i++) R.arr[i]=arr[i].real();  
      return R; 
    }

    RscalarA* realp() const{
      RscalarA* R=new RscalarA(nbu,fill::raw);
      if(nbu==-1) R->val=val.real();
      else for(int i=0; i<nbu; i++) R->arr[i]=arr[i].real();  
      return R; 
    }

    RscalarA imag() const{
      RscalarA R(nbu,fill::raw);
      if(nbu==-1) R.val=val.imag();
      else for(int i=0; i<nbu; i++) R.arr[i]=arr[i].imag();  
      return R; 
    }

    RscalarA* imagp() const{
      RscalarA* R=new RscalarA(nbu,fill::raw);
      if(nbu==-1) R->val=val.imag();
      else for(int i=0; i<nbu; i++) R->arr[i]=arr[i].imag();  
      return R; 
    }

    void set_real(const RscalarA& x){
      if(nbu==-1) val.real(x.val);
      else for(int i=0; i<nbu; i++) arr[i].real(x.arr[i]);
    }

    void set_imag(const RscalarA& x){
      if(nbu==-1) val.imag(x.val);
      else for(int i=0; i<nbu; i++) arr[i].imag(x.arr[i]);
    }

    bool operator==(const CscalarA& x) const{
      assert(nbu==x.nbu);
      if(nbu==-1) return val==x.val;
      for(int i=0; i<nbu; i++)
	if(arr[i]!=x.arr[i]) return false;
      return true;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    void zero(){
      if(nbu==-1) val=0;
      else std::fill(arr, arr+nbu,0);
    }
    
    void set_zero(){
      if(nbu==-1) val=0;
      else std::fill(arr, arr+nbu,0);
    }

    CscalarA conj() const{
      CscalarA R(nbu,fill::raw);
      if(nbu==-1) R.val=std::conj(val);
      else for(int i=0; i<nbu; i++) R.arr[i]=std::conj(arr[i]);
      return R;
    }

    CscalarA* conjp() const{
      CscalarA* R=new CscalarA(nbu,fill::raw);
      if(nbu==-1) R->val=std::conj(val);
      else for(int i=0; i<nbu; i++) R->arr[i]=std::conj(arr[i]);
      return R;
    }

    static CscalarA* sum(const vector<CscalarA*> v){
      const int N=v.size();
      if(N==0) return new CscalarA(0);
      const int nbu=v[0]->nbu;
      if(nbu==-1){
	complex<float> s=0;
	for(int i=0; i<N; i++)
	  s+=v[i]->val;
	return new CscalarA(s);
      }else{
	CNINE_UNIMPL();
	return new CscalarA(0);
      }
    }

    CscalarA apply(std::function<complex<float>(const complex<float>)> fn){
      if(nbu==-1) return CscalarA(fn(val));
      CscalarA R(nbu,fill::raw);
      for(int i=0; i<nbu; i++)
	R.arr[i]=fn(arr[i]);
      return R;
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------

    
    void add(const CscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i];
    }

    void add(const CscalarA& x, const float c){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=c*x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=c*x.arr[i];
    }

    void add(const CscalarA& x, const complex<float> c){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=c*x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=c*x.arr[i];
    }

    void add_to_real(const RscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=complex<float>(x.val,0);
      else for(int i=0; i<nbu; i++) arr[i]+=complex<float>(x.arr[i],0);
    }

    void add_to_imag(const RscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=complex<float>(0,x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=complex<float>(0,x.arr[i]);
    }

    void add_conj(const CscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=std::conj(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::conj(x.arr[i]);
    }

    void add_real_to(RscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) x.val+=val.real();
      else for(int i=0; i<nbu; i++) x.arr[i]+=arr[i].real();
    }

    void add_imag_to(RscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) x.val+=val.imag();
      else for(int i=0; i<nbu; i++) x.arr[i]+=arr[i].imag();
    }

    void add_sum(const vector<CscalarA*> v){
      const int N=v.size();
      if(N==0) return; 
      const int nbu=v[0]->nbu;
      if(nbu==-1){
	for(int i=0; i<N; i++)
	  val+=v[i]->val;
      }else{
	CNINE_UNIMPL();
      }
    }


    void subtract(const CscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val-=x.val;
      else for(int i=0; i<nbu; i++) arr[i]-=x.arr[i];
    }

    void add_minus(const CscalarA& x, const CscalarA& y){
      assert(nbu==x.nbu);
      assert(nbu==y.nbu);
      if(nbu==-1) val+=x.val-y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]-y.arr[i];
    }

    void add_prod(const CscalarA& x, const CscalarA& y){
      if(nbu==-1) val+=x.val*y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*y.arr[i];
    }

    void add_prodc(const CscalarA& x, const CscalarA& y){
      if(nbu==-1) val+=x.val*std::conj(y.val);
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*std::conj(y.arr[i]);
    }

    void add_prodc1(const CscalarA& x, const CscalarA& y){
      if(nbu==-1) val+=x.val*std::conj(y.val);
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*std::conj(y.arr[i]);
    }

    void add_prodcc(const CscalarA& x, const CscalarA& y){
      if(nbu==-1) val+=std::conj(x.val*y.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::conj(x.arr[i]*y.arr[i]);
    }

    void add_prod(const CscalarA& x, const RscalarA& y){
      if(nbu==-1) val+=x.val*y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*y.arr[i];
    }

    void add_div(const CscalarA& x, const CscalarA& y){
      if(nbu==-1) val+=x.val/y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]/y.arr[i];
    }

    void add_div_back0(const CscalarA& x, const CscalarA& y){
      if(nbu==-1) val+=x.val/std::conj(y.val); 
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]/std::conj(y.arr[i]);
    }
 
    void add_div_back1(const CscalarA& g, const CscalarA& x, const CscalarA& y){
      if(nbu==-1) val-=g.val*std::conj(x.val*complex<float>(pow(y.val,-2.0)));
      else for(int i=0; i<nbu; i++) arr[i]-=g.arr[i]*std::conj(x.arr[i]*complex<float>(pow(y.arr[i],-2.0)));
    }

    void add_norm2_to(RscalarA& r) const{
      assert(nbu==r.nbu);
      if(nbu==-1) r.val+=std::real(val*std::conj(val));
      else for(int i=0; i<nbu; i++) r.arr[i]+=std::real(arr[i]*std::conj(arr[i]));
    }

    void add_norm2_back(const RscalarA& x, const CscalarA& y){
      assert(nbu==x.nbu);
      assert(nbu==y.nbu);
      if(nbu==-1) val+=x.val*(y.val+std::conj(y.val));
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*(y.arr[i]+std::conj(y.arr[i]));
    }

    void add_abs(const CscalarA& x){
      if(nbu==-1) val+=std::abs(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::abs(x.arr[i]);
    }

    void add_abs_back(const CscalarA& g, const CscalarA& x){
      if(nbu==-1) val+=complex<float>(ifthen(x.val.real()>0,g.val.real(),-g.val.real()),
	ifthen(x.val.imag()>0,g.val.imag(),-g.val.imag()));
      else for(int i=0; i<nbu; i++){
	float re=g.arr[i].real();
	float im=g.arr[i].imag();
	arr[i]+=complex<float>(ifthen(x.arr[i].real()>0,re,-re),ifthen(x.arr[i].imag()>0,im,-im));
      }
    }

    void add_pow(const CscalarA& x, const float p, const complex<float> c=1.0){
      if(nbu==-1) val+=c*std::pow(x.val,p);
      else for(int i=0; i<nbu; i++) arr[i]+=c*std::pow(x.arr[i],p);
    }

    void add_pow_back(const CscalarA& g, const CscalarA& x, const float p, const complex<float> c=1.0){
      if(nbu==-1) val+=c*g.val*std::conj(std::pow(x.val,p));
      else for(int i=0; i<nbu; i++) arr[i]+=c*g.arr[i]*std::conj(std::pow(x.arr[i],p));
    }

    void add_exp(const CscalarA& x){
      if(nbu==-1) val+=std::exp(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::exp(x.arr[i]);
    }

    void add_ReLU(const CscalarA& x, const float c){
      if(nbu==-1){
	float re=x.val.real();
	float im=x.val.imag();
	val+=complex<float>(ifthen(re>0,re,c*re),ifthen(im>0,im,c*im));
      }
      else{
	for(int i=0; i<nbu; i++){
	  float re=x.arr[i].real();
	  float im=x.arr[i].imag();
	  arr[i]+=complex<float>(ifthen(re>0,re,c*re),ifthen(im>0,im,c*im));
	}
      }
    }

    void add_ReLU_back(const CscalarA& g, const CscalarA& x, const float c){
      if(nbu==-1){
	float re=g.val.real();
	float im=g.val.imag();
	val+=complex<float>(ifthen(x.val.real()>0,re,c*re),ifthen(x.val.imag()>0,im,c*im));
      }
      else{
	for(int i=0; i<nbu; i++){
	  float re=g.arr[i].real();
	  float im=g.arr[i].imag();
	  arr[i]+=complex<float>(ifthen(x.arr[i].real()>0,re,c*re),ifthen(x.arr[i].imag()>0,im,c*im));
	}
      }
    }

    void add_sigmoid(const CscalarA& x){
      if(nbu==-1){
	val+=complex<float>(1.0/(1.0+std::exp(-x.val.real())),1.0/(1.0+std::exp(-x.val.imag())));
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=complex<float>(1.0/(1.0+std::exp(-x.arr[i].real())),1.0/(1.0+std::exp(-x.arr[i].imag())));
	}
      }
    }

    void add_sigmoid_back(const CscalarA& g, const CscalarA& x){
      if(nbu==-1){
	val+=complex<float>(x.val.real()*(1.0-x.val.real())*g.val.real(),x.val.imag()*(1.0-x.val.imag())*g.val.imag());
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=complex<float>(x.arr[i].real()*(1.0-x.arr[i].real())*g.arr[i].real(),
	    x.arr[i].imag()*(1.0-x.arr[i].imag())*g.arr[i].imag());
	}
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    /*
    CscalarA(const string filename, const device_id& dev=0):
      CFscalar(filename,dev){}

    int save(const string filename) const{
      CFscalar::save(filename);
      return 0;
    }

    CscalarA(Bifstream& ifs): 
      CFscalar(ifs){
    }

    void serialize(Bofstream& ofs){
      CFscalar::serialize(ofs);
    }
    */

    string str(const string indent="") const{
      ostringstream oss;
      if(nbu==-1){
	oss<<std::real(val);
	return oss.str();
      }
      oss<<"[ ";
      for(int i=0; i<nbu; i++)
	oss<<arr[i]<<" ";
      oss<<"]";
      return oss.str();
    }
   
  };


  /*
  inline CscalarA& asCscalarA(Cobject* x, const char* s){
    return downcast<CscalarA>(x,s);
  }

  inline CscalarA& asCscalarA(Cnode* x, const char* s){
    return downcast<CscalarA>(x,s);
  }
  
  inline CscalarA& asCscalarA(Cnode& x, const char* s){
    return downcast<CscalarA>(x,s);
  }
  */

}


// #define CSCALARB(x) asCscalarA(x,__PRETTY_FUNCTION__) 


#endif


