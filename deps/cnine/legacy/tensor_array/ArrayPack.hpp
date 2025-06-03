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


#ifndef _ArrayPack
#define _ArrayPack

#include "CtensorArray.hpp"

namespace cnine{

  template<typename ARRAY>
  class ArrayPack{
  public:

    Gdims adims;

    vector<ARRAY*> array;

    ~ArrayPack(){
      for(auto p:array)
	delete p;
    }


  public: // ---- Constructors -------------------------------------------------------------------------------

    
    ArrayPack(const Gdims& _adims):
      adims(_adims){}

    
  public: // ---- Copying ------------------------------------------------------------------------------------

    
    ArrayPack(const ArrayPack& x):
      adims(x.adims){
      for(auto p: x.array)
	array.push_back(new ARRAY(*p));
    }
    
    ArrayPack(ArrayPack&& x):
      adims(x.adims){
      for(auto p: x.array)
	array.push_back(new ARRAY(std::move(*p)));
    }

    ArrayPack& operator=(const ArrayPack& x){
      adims=x.adims;
      for(auto p:array)
	delete p;
      array.clear();
      for(auto p:x.array)
	array.push_back(new ARRAY(*p));
      return *this;
    }
    
    ArrayPack& operator=(ArrayPack&& x){
      adims=x.adims;
      for(auto p:array)
	delete p;
      array.clear();
      array=x.array;
      x.array.clear();
      return *this;
    }
    

  public: // ---- Variants -----------------------------------------------------------------------------------


    ArrayPack(const ArrayPack& x, const int _dev):
      adims(x.adims){
      for(auto p:x.array)
	array.push_back(new ARRAY(*p,_dev));
    }

    ArrayPack(const ArrayPack& x, const view_flag& flag):
      adims(x.adims){
      for(auto p:x.array)
	array.push_back(new ARRAY(*p,flag));
    }

   template<typename FILLTYPE, typename = 
	    typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
   ArrayPack(const ArrayPack& x, const FILLTYPE& fill):
     adims(x.adims){
     for(auto p:x.array)
       array.push_back(new ARRAY(*p,fill));
   }
	 

  public: // ---- Transport ----------------------------------------------------------------------------------


    ArrayPack& to_device(const int _dev){
      for(auto p:array)
	p->to_device(_dev);
      return *this;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int get_dev() const{
      if(array.size()==0) return 0;
      return array[0]->get_dev();
    }
    
    int get_nbu() const{ 
      if(array.size()==0) return -1;
      return array[0]->get_nbu();
    }

    int get_adim(const int i) const{
      return adims[i];
    }

    Gdims get_adims() const{
      return adims;
    }

    int get_aasize() const{
     if(array.size()==0) return 0;
      return array[0]->aasize;
    }
      

    ARRAY get_array(const int l) const{
      assert(l<array.size() && array[l]);
      return ARRAY(*array[l]);
    }

    void set_array(const int l, const ARRAY& x){
      assert(l<array.size());
      if(array[l]) (*array[l])=x;
      else array[l]=new ARRAY(x);
    }


  public: // ---- Broadcasting and reductions ----------------------------------------------------------------


    template<typename PACK>
    void broadcast_copy(const PACK& x){
      for(int i=0; i<array.size(); i++)
	array[i]->broadcast_copy(*x.partp(i));
    }

    void broadcast_copy(const int ix, const ArrayPack& x){
      for(int i=0; i<array.size(); i++)
	array[i]->broadcast_copy(ix,*x.array[i]);
    }

    void add_reduce(const ArrayPack& x, const int ix){
      assert(array.size()==x.array.size());
      for(int i=0; i<array.size(); i++)
	array[i]->add_reduce(*x.array[i],ix);
    }


  public: // ---- Cumulative operations -----------------------------------------------------------------------


    void add(const ArrayPack& y){
      assert(y.array.size()==array.size());
      for(int i=0; i<array.size(); i++)
	array[i]->add(*y.array[i]);
    }
    
    void subtract(const ArrayPack& y){
      assert(y.array.size()==array.size());
      for(int i=0; i<array.size(); i++)
	array[i]->subtract(*y.array[i]);
    }

    void add(const ArrayPack& y, const CscalarObj& c){
      assert(y.array.size()==array.size());
      for(int i=0; i<array.size(); i++)
	array[i]->add(*y.array[i],c);
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    ArrayPack plus(const ArrayPack& y) const{
      ArrayPack R(adims);
      for(int i=0; i<array.size(); i++)
	R.array.push_back(new ARRAY(array[i]->plus(*y.array[i])));
      return R;
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      ostringstream oss;
      for(auto p: array)
	oss<<p->str(indent)<<endl;
      return oss.str();
    }

  };

}

#endif
    //template<typename FILLTYPE, typename = typename std::enable_if<std::is_base_of<fill_pattern, FILLTYPE>::value, FILLTYPE>::type>
    //ObjectArray(const Gdims& _adims, const SO3type& _tau, const int _nbu=-1, const FILLTYPE& fill, const int _dev=0):
    //tau(_tau){
      
    //}
