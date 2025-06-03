/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _multivec
#define _multivec

#include "Cnine_base.hpp"
#include "GenericIterator.hpp"


namespace cnine{


  template<typename TYPE>
  class multivec{
  public:

    vector<vector<TYPE> > vecs;

    class iterator: public GenericIterator<multivec,vector<TYPE> >{
    public:
      using BASE=GenericIterator<multivec,vector<TYPE> >;
      using BASE::BASE;
    };

    class citerator: public GenericConstIterator<multivec,vector<TYPE> >{
    public:
      using BASE=GenericConstIterator<multivec,vector<TYPE> >;
      using BASE::BASE;
    };


  public: // ---- Constructors ------------------------------------------------------------------------------


    multivec(){}

    multivec(const int k){
      for(int i=0; i<k; i++)
	vecs.push_back(vector<TYPE>());
    }

    multivec(const int k, const int n, const int fcode=0):
      multivec(k){
      if(fcode==3){
	int z=0;
	for(int i=0; i<n; i++){
	  vector<TYPE> v;
	  for(int j=0; j<k; j++) v.push_back(z++);
	  push_back(v);
	}
      }
    }


  public: // ---- Access ------------------------------------------------------------------------------------


    int nvecs() const{
      return vecs.size();
    }

    size_t size() const{
      if(vecs.size()==0) return 0;
      return vecs[0].size();
    }

    void check_size(const int k){
      if(nvecs()==0)
	for(int i=0; i<k; i++)
	  vecs.push_back(vector<TYPE>());
      CNINE_ASSRT(nvecs()==k);
    }

    void push_back(const initializer_list<TYPE>& list){
      check_size(list.size());
      //CNINE_ASSRT(list.size()==vecs.size());
      int i=0;
      for(auto& p:list)
	vecs[i++].push_back(p);
    }

   void push_back(const vector<TYPE>& list){
      check_size(list.size());
      //CNINE_ASSRT(list.size()==vecs.size());
      int i=0;
      for(auto& p:list)
	vecs[i++].push_back(p);
    }

    vector<TYPE> operator[](const int i) const{
      vector<TYPE> r(nvecs());
      for(int j=0; j<nvecs(); j++)
	r[j]=vecs[j][i];
      return r;
    }
    
    citerator begin() const{
      return citerator(this,0);
    }

    citerator end() const{
      return citerator(this,size());
    }

    iterator begin(){
      return iterator(this,0);
    }

    iterator end(){
      return iterator(this,size());
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      for(auto p:*this){
	oss<<p<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const multivec& x){
      stream<<x.str(); return stream;
    }

  };

}

#endif 
  /*
  template<typename T, std::size_t N>
  struct homogeneous_tuple {
    using type=decltype(std::tuple_cat(std::tuple<T>{},
	typename homogeneous_tuple<T,N-1>::type{}));
  };
  
  template<typename T>
  struct homogeneous_tuple<T, 0> {
    using type = std::tuple<>;
  };

  template<typename T, std::size_t N>
  using homogeneous_tuple_t = typename homogeneous_tuple<T,N>::type;


  template<typename T, std::size_t N>
  struct n_vectors_helper{
    using type=decltype(std::tuple_cat(std::tuple<vector<T> >{}, typename n_vectors_helper<T,N-1>::type{}));};
  
  template<typename T>
  struct n_vectors_helper<T,0> {using type=std::tuple<>;};

  template<typename T, std::size_t N>
  using n_vectors_t = typename n_vectors_helper<T,N>::type;
  */
