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

#ifndef _CSRvector
#define _CSRvector

#include "Cnine_base.hpp"
#include "array_pool.hpp"


namespace cnine{

  template<typename TYPE>
  class CSRvector{
  public:

    TYPE* arr=nullptr;
    int n=0;
    bool is_view=false;

    ~CSRvector<TYPE>(){
      if(is_view) return;
      delete[] arr;
    }

  public: // ---- Constructors -------------------------------------------------------------------------------

    
    CSRvector(const int _n): n(_n){
      arr=new int[2*n];
    }
    CSRvector(const int _n, cnine::fill_zero& dummy): n(_n){
      arr=new int[2*n];
      for(int i=0; i<n; i++){
	arr[2*i]=reinterpret_cast<TYPE>(&i);
	arr[2*i+1]=0;
      }
    }

    CSRvector(float* _arr, const int _n):
      arr(_arr), n(_n), is_view(true){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    CSRvector(const CSRvector<TYPE>& x){
      n=x.n;
      arr=new TYPE[2*n];
      std::copy(x.arr,x.arr+2*n,arr);
    }

    CSRvector(CSRvector<TYPE>&& x){
      n=x.n; x.n=0;
      arr=x.arr; x.arr=nullptr;
      is_view=x.is_view;
    }

    CSRvector<TYPE>& operator=(const CSRvector<TYPE>& x)=delete;
    

  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return n;
    }

    pair<int,TYPE> operator[](const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In CSRvector<TYPE>::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return pair<int,float>(reinterpret_cast<int>(arr[2*i]),arr[2*i+1]);
    }

    int ix(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In CSRvector<TYPE>::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return reinterpret_cast<int>(arr[2*i]);
    }

    int val(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In CSRvector<TYPE>::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return arr[2*i+1];
    }

    void for_each(std::function<void(const int i, const TYPE val)> lambda) const{
      for(int i=0; i<n; i++)
	lambda(*reinterpret_cast<int*>(arr+2*i),arr[2*i+1]);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------

    
    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"(";
      for(int i=0; i<n; i++){
	oss<<"("<<*reinterpret_cast<int*>(arr+2*i)<<","<<arr[2*i+1]<<")";
	if(i<n-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }
    
    friend ostream& operator<<(ostream& stream, const CSRvector<TYPE>& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
