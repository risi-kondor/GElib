/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _auto_array
#define _auto_array

#include "Cnine_base.hpp"


namespace cnine{

  template<typename TYPE>
  class auto_array{
  public:

    typedef std::size_t size_t;

    mutable TYPE* arr;
    mutable size_t memsize;
    mutable size_t _size;

    ~auto_array(){
      if(arr) delete[] arr;
    }

  public: //---- Constructors -------------------------------------


    auto_array(){
      arr=new TYPE[1];
      memsize=1;
      _size=0;
    }

    auto_array(const int n){
      arr=new TYPE[n];
      memsize=n;
      _size=n;
    }


  public: //---- Copying -------------------------------------


    auto_array(const auto_array& x):
      memsize(x.memsize),
      _size(x._size){
      arr=new int[memsize];
    }

    auto_array(auto_array&& x):
      memsize(x.memsize),
      _size(x._size){
      arr=x.arr;
      x.arr=nullptr;
    }


  public: //---- Resizing -------------------------------------


    void resize(const size_t n) const{
      if(memsize<n) reserve(n);
      _size=n;
    }

    void reserve(const size_t x) const{
      if(x<=memsize) return;
      size_t new_memsize=std::max(x,2*memsize);
      int* newarr=new TYPE[new_memsize];
      std::copy(arr,arr+memsize,newarr);
      delete[] arr;
      arr=newarr;
      memsize=new_memsize;
    }


  public: //---- Access -------------------------------------


    size_t size() const{
      return _size;
    }

    TYPE operator[](const size_t i) const{
      if(i>=_size) resize(i+1);
      return arr[i];
    }

    TYPE& operator[](const size_t i){
      if(i>=_size) resize(i+1);
      return arr[i];
    }

    int get(const size_t i) const{
      if(i>=_size) resize(i+1);
      return arr[i];
    }

    void set(const size_t i, const TYPE& x){
      if(i>=_size) resize(i+1);
      arr[i]=x;
    }


  public: //---- I/O -- -------------------------------------

    
    string str(){
      ostringstream oss;
      oss<<"[";
      for(int i=0; i<size()-1; i++)
	oss<<arr[i]<<",";
      if(_size>0) oss<<arr[_size-1];
      oss<<"]";
      return oss.str();
    }

  };


}

#endif 
