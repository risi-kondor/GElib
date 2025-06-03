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


#ifndef _CnineRscalarA
#define _CnineRscalarA

#include "Cnine_base.hpp"
#include "CnineObject.hpp"

extern default_random_engine rndGen;


namespace cnine{

  class RscalarA: public CnineObject{ //, public CnineBackendObject{
  public:

    int nbu=-1;
    float val;
    float* arr=nullptr;


    RscalarA(){
    }

    ~RscalarA(){
      delete[] arr;
    }

    string classname() const{
      return "RscalarA";
    }

    string describe() const{
      if(nbu>=0) return "RscalarA["+to_string(nbu)+"]";
      return "RscalarA";
    }


  private: // ---- Private Constructors ---------------------------------------------------------------------


    void reallocate(){
      delete[] arr;
      if(nbu==-1) return;
      arr=new float[nbu];
    }


  public: // ---- Filled constructors -----------------------------------------------------------------------


    RscalarA(const fill_raw& fill){}

    RscalarA(const fill_zero& fill): val(0){}
 
    RscalarA(const fill_ones& fill): val(1){}

    RscalarA(const float c): val(c){}
 
    RscalarA(const double c): val(c){}
 
    RscalarA(const fill_gaussian& fill){
      normal_distribution<float> distr;
      val=distr(rndGen);
    }

    RscalarA(const fill_gaussian& fill, const float c){
      normal_distribution<float> distr;
      val=c*distr(rndGen);
    }

    RscalarA(const int _nbu, const fill_raw& fill): 
      nbu(_nbu){
      reallocate();
    }

    RscalarA(const int _nbu, const fill_zero& fill): 
      RscalarA(_nbu,fill::raw){
      if(nbu==-1) val=0; 
      else std::fill(arr,arr+nbu,0);
    }
 
    RscalarA(const int _nbu, const fill_gaussian& fill):
      RscalarA(_nbu,fill::raw){
      normal_distribution<float> distr;
      if(nbu==-1) val=distr(rndGen);
      else for(int i=0; i<nbu; i++) arr[i]=distr(rndGen);
    }

    RscalarA(const int _nbu, const fill_gaussian& fill, const float c):
      RscalarA(_nbu,fill::raw){
      normal_distribution<float> distr;
      if(nbu==-1) val=c*distr(rndGen);
      else for(int i=0; i<nbu; i++) arr[i]=c*distr(rndGen);
    }

    RscalarA(const int _nbu, const float c): 
      RscalarA(_nbu,fill::raw){
      if(nbu==-1) val=c; 
      else std::fill(arr,arr+nbu,c);
    }
 
    RscalarA(const RscalarA& x, std::function<float(const float)> fn):
      RscalarA(x.nbu,fill::raw){
      if(nbu==-1) val=fn(x.val); 
      else for(int i=0; i<nbu; i++) arr[i]=fn(x.arr[i]);
    }


  public: // ---- Copying -----------------------------------------------------------------------------------


    RscalarA(const RscalarA& x): 
      nbu(x.nbu){
      if(nbu==-1) val=x.val;
      else{
	reallocate();
	std::copy(x.arr,x.arr+nbu,arr);
      }
    }

    RscalarA(RscalarA&& x): 
      nbu(x.nbu){
      if(nbu==-1) val=x.val;
      else{
	arr=x.arr;
	x.arr=nullptr;
      }
    }

    RscalarA& operator=(const RscalarA& x){
      delete[] arr;
      arr=nullptr;
      nbu=x.nbu;
      if(nbu==-1){
	val=x.val;
      }else{
	arr=new float[nbu];
	std::copy(x.arr,x.arr+nbu,arr);
      }
      return *this;
    }
    
    RscalarA& operator=(RscalarA&& x){
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

    //CnineObject* spawn_zero() const{
    //return new RscalarA(nbu,fill::zero);
    //}


  public: // ---- Conversions -------------------------------------------------------------------------------


    RscalarA(const vector<float>& v){
      if(v.size()==1){
	nbu=-1;
	val=v[0];
      }else{
	nbu=v.size();
	arr=new float[nbu];
	for(int i=0; i<nbu; i++) 
	  arr[i]=v[i];
      }
    }

    operator vector<float>(){
      if(nbu==-1){return vector<float>(1,val);}
      vector<float> R(nbu);
      for(int i=0; i<nbu; i++)
	R[i]=arr[i];
      return R;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int getnbu() const{
      return nbu;
    }

    float get_value() const{
      assert(nbu==-1);
      return val;
    }

    float value() const{
      assert(nbu==-1);
      return val;
    }

    explicit operator float() const{
      assert(nbu==-1);
      return val;
    }

    RscalarA& set_value(float x){
      assert(nbu==-1);
      val=x;
      return *this;
    }
    
    RscalarA& set(float x){
      assert(nbu==-1);
      val=x;
      return *this;
    }

    RscalarA& operator=(float x){
      assert(nbu==-1);
      val=x;
      return *this;
    }

    bool operator==(const RscalarA& x) const{
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

    RscalarA* conj() const{
      RscalarA* R=new RscalarA(nbu,fill::raw);
      if(nbu==-1) R->val=val;
      else for(int i=0; i<nbu; i++) R->arr[i]=arr[i];
      return R;
    }

    RscalarA apply(std::function<float(const float)> fn){
      if(nbu==-1) return RscalarA(fn(val));
      RscalarA R(nbu,fill::raw);
      for(int i=0; i<nbu; i++)
	R.arr[i]=fn(arr[i]);
      return R;
    }


  public: // ---- Cumulative Operations ----------------------------------------------------------------------

    
    void add(const RscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i];
    }

    void add_conj(const RscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i];
    }

    void add(const RscalarA& x, const float c){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=c*x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=c*x.arr[i];
    }

    void add_sum(const vector<RscalarA*> v){
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

    void subtract(const RscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val-=x.val;
      else for(int i=0; i<nbu; i++) arr[i]-=x.arr[i];
    }

    void add_minus(const RscalarA& x, const RscalarA& y){
      assert(nbu==x.nbu);
      assert(nbu==y.nbu);
      if(nbu==-1) val+=x.val-y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]-y.arr[i];
    }

    void add_prod(const RscalarA& x, const RscalarA& y){
      assert(nbu==x.nbu);
      assert(nbu==y.nbu);
      if(nbu==-1) val+=x.val*y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*y.arr[i];
    }

    void add_prodc1(const RscalarA& x, const RscalarA& y){
      assert(nbu==x.nbu);
      assert(nbu==y.nbu);
      if(nbu==-1) val+=x.val*y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*y.arr[i];
    }

    void add_div(const RscalarA& x, const RscalarA& y){
      assert(nbu==x.nbu);
      assert(nbu==y.nbu);
      if(nbu==-1) val+=x.val/y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]/y.arr[i];
    }

    void add_div_back0(const RscalarA& x, const RscalarA& y){
      assert(nbu==x.nbu);
      assert(nbu==y.nbu);
      if(nbu==-1) val+=x.val/y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]/y.arr[i];
    }

    void add_div_back1(const RscalarA& g, const RscalarA& x, const RscalarA& y){
      assert(nbu==g.nbu);
      assert(nbu==x.nbu);
      assert(nbu==y.nbu);
      if(nbu==-1) val-=g.val*x.val/y.val/y.val;
      else for(int i=0; i<nbu; i++) arr[i]-=g.arr[i]*x.arr[i]*pow(y.arr[i],-2.0);
    }

    void add_norm2(const RscalarA& x){
      assert(nbu==x.nbu);
      if(nbu==-1) val+=x.val*x.val;
      else for(int i=0; i<nbu; i++) arr[i]+=x.arr[i]*x.arr[i];
    }

    void add_norm2_to(RscalarA& r) const{
      assert(nbu==r.nbu);
      if(nbu==-1) r.val+=val*val;
      else for(int i=0; i<nbu; i++) r.arr[i]+=arr[i]*arr[i];
    }

    void add_norm2_back(const RscalarA& x, const RscalarA& y){
      assert(nbu==x.nbu);
      assert(nbu==y.nbu);
      if(nbu==-1) val+=2.0*x.val*y.val;
      else for(int i=0; i<nbu; i++) arr[i]+=2.0*x.arr[i]*y.arr[i];
    }

    void add_abs(const RscalarA& x){
      if(nbu==-1) val+=std::abs(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::abs(x.arr[i]);
    }

    void add_abs_back(const RscalarA& g, const RscalarA& x){
      if(nbu==-1) val+=ifthen(x.val>0,g.val,-g.val);
      else for(int i=0; i<nbu; i++) arr[i]+=ifthen(x.arr[i]>0,g.arr[i],-g.arr[i]);
    }

    void add_pow(const RscalarA& x, const float p, const float c=1.0){
      if(nbu==-1) val+=c*std::pow(x.val,p);
      else for(int i=0; i<nbu; i++) arr[i]+=c*std::pow(x.arr[i],p);
    }

    void add_pow_back(const RscalarA& g, const RscalarA& x, const float p, const float c=1.0){
      if(nbu==-1) val+=c*g.val*std::pow(x.val,p);
      else for(int i=0; i<nbu; i++) arr[i]+=c*g.arr[i]*std::pow(x.arr[i],p);
    }

    void add_exp(const RscalarA& x){
      if(nbu==-1) val+=std::exp(x.val);
      else for(int i=0; i<nbu; i++) arr[i]+=std::exp(x.arr[i]);
    }

    void add_ReLU(const RscalarA& x, const float c){
      if(nbu==-1){
	val+=ifthen(x.val>0,x.val,c*x.val);
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=ifthen(x.arr[i]>0,x.arr[i],c*x.arr[i]);
	}
      }
    }

    void add_ReLU_back(const RscalarA& g, const RscalarA& x, const float c){
      if(nbu==-1){
	val+=ifthen(x.val>0,g.val,c*g.val);
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=ifthen(x.arr[i]>0,g.arr[i],c*g.arr[i]);
	}
      }
    }

    void add_sigmoid(const RscalarA& x){
      if(nbu==-1){
	val+=1.0/(1.0+std::exp(-x.val));
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=1.0/(1.0+std::exp(-x.arr[i]));
	}
      }
    }

    void add_sigmoid_back(const RscalarA& g, const RscalarA& x){
      if(nbu==-1){
	val+=x.val*(1.0-x.val)*g.val;
      }
      else{
	for(int i=0; i<nbu; i++){
	  arr[i]+=x.arr[i]*(1.0-x.arr[i])*g.arr[i];
	}
      }
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string str(const string indent="") const{
      if(nbu==-1) return to_string(val);
      ostringstream oss;
      oss<<"[ ";
      for(int i=0; i<nbu; i++)
	oss<<arr[i]<<" ";
      oss<<"]";
      return oss.str();
    }
    

   
  };



  

}


#endif



  /*
  inline RscalarA& asRscalarA(Cobject* x){
    assert(x); 
    if(!dynamic_cast<RscalarA*>(x))
      cerr<<"Cengine error: Cobject is of type "<<x->classname()<<" instead of RscalarA."<<endl;
    assert(dynamic_cast<RscalarA*>(x));
    return static_cast<RscalarA&>(*x);
  }

  inline RscalarA& asRscalarA(Cnode* x){
    assert(x->obj);
    if(!dynamic_cast<RscalarA*>(x->obj))
      cerr<<"Cengine error: Cobject is of type "<<x->obj->classname()<<" instead of RscalarA."<<endl;
    assert(dynamic_cast<RscalarA*>(x->obj));
    return static_cast<RscalarA&>(*x->obj);
  }

  inline RscalarA& asRscalarA(Cnode& x){
    assert(x.obj);
    if(!dynamic_cast<RscalarA*>(x.obj))
      cerr<<"Cengine error: Cobject is of type "<<x.obj->classname()<<" instead of RscalarA."<<endl;
    assert(dynamic_cast<RscalarA*>(x.obj));
    return static_cast<RscalarA&>(*x.obj);
  }
  */
    /*
    RscalarA(const string filename, const device_id& dev=0):
      CFscalar(filename,dev){}

    int save(const string filename) const{
      CFscalar::save(filename);
      return 0;
    }

    RscalarA(Bifstream& ifs): 
      CFscalar(ifs){
    }

    void serialize(Bofstream& ofs){
      CFscalar::serialize(ofs);
    }
    */

  //inline RscalarA& asRscalarA(Cnode* x){
  //return downcast<RscalarB>(x,"");
  //}

  //inline RscalarB& asRscalarB(Cnode& x){
  //return downcast<RscalarB>(x,"");
  //}

