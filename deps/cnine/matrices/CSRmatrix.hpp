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

#ifndef _CSRmatrix
#define _CSRmatrix

#include "Cnine_base.hpp"
//#include "RtensorA.hpp"
#include "Tensor.hpp"
#include "array_pool.hpp"
#include "CSRvector.hpp"


namespace cnine{


  template<class TYPE>
  class CSRmatrix: public array_pool<TYPE>{
  public:

    using array_pool<TYPE>::arr;
    using array_pool<TYPE>::arrg;
    using array_pool<TYPE>::tail;
    using array_pool<TYPE>::memsize;
    using array_pool<TYPE>::dev;
    using array_pool<TYPE>::is_view;
    using array_pool<TYPE>::dir;

    using array_pool<TYPE>::reserve;
    using array_pool<TYPE>::size;
    using array_pool<TYPE>::offset;
    using array_pool<TYPE>::size_of;

    using array_pool<TYPE>::get_device;

    int n=0;
    int m=0;

    mutable CSRmatrix<TYPE>* transpp=nullptr;

    ~CSRmatrix(){
      if(is_view) return;
      if(transpp) delete transpp;
    }
      

  public: // ---- Constructors ------------------------------------------------------------------------------------


    CSRmatrix():
      array_pool<TYPE>(){}

    CSRmatrix(const int _n, const int _m):
      array_pool<TYPE>(_n), n(_n), m(_m){}


  public: // ---- Copying ------------------------------------------------------------------------------------


    CSRmatrix(const CSRmatrix<TYPE>& x):
      array_pool<TYPE>(x){
      n=x.n;
      m=x.m;
    }

    CSRmatrix(CSRmatrix<TYPE>&& x):
      array_pool<TYPE>(std::move(x)){
      n=x.n;
      m=x.m;
    }

    CSRmatrix& operator=(const CSRmatrix<TYPE>& x){
      array_pool<TYPE>::operator=(x);
      n=x.n;
      m=x.m;
      return *this;
    }

  
  public: // ---- Transport ----------------------------------------------------------------------------------


    CSRmatrix(const CSRmatrix& x, const int _dev):
      array_pool<float>(x,_dev), n(x.n), m(x.m){
    }

    CSRmatrix& to_device(const int _dev){
      array_pool<float>::to_device(_dev);
      return *this;
    }


  public: // ---- Conversions --------------------------------------------------------------------------------


    CSRmatrix(const TensorView<TYPE>& x):
      CSRmatrix(x.view2()){}

    CSRmatrix(const Rtensor2_view& x):
      CSRmatrix(x.n0,x.n1){
      dir.resize0(x.n0);

      int t=0;
      for(int i=0; i<n; i++)
	for(int j=0; j<m; j++)
	  if(x(i,j)!=0) t++;
      array_pool<TYPE>::reserve(2*t);

      tail=0;
      for(int i=0; i<n; i++){
	dir.set(i,0,tail);
	int m=0;
	for(int j=0; j<x.n1; j++)
	  if(x(i,j)!=0){
	    *reinterpret_cast<int*>(arr+tail+2*m)=j;
	    arr[tail+2*m+1]=x(i,j);
	    m++;
	  }
	dir.set(i,1,2*m);
	tail+=2*m;
      }
    }

    operator TensorView<TYPE>() const{
      TensorView<TYPE> R({n,m},0,0);
      for_each([&](const int i, const int j, const float v){
	  R.set(i,j,v);
	});
      return R;
    }

//     operator RtensorA() const{
//       RtensorA R=RtensorA::zero({n,m});
//       for_each([&](const int i, const int j, const float v){
// 	  R.set(i,j,v);
// 	});
//       return R;
//     }


  public: // ---- ATEN ---------------------------------------------------------------------------------------


#ifdef _WITH_ATEN

    CSRmatrix(const at::Tensor& T){
      CNINE_CONVERT_FROM_ATEN_WARNING();
      CNINE_ASSERT(T.dim()==2,"Number of dimensions of tensor to be converted to CSRmatrix must be 2");
      (*this)=CSRmatrix(Tensor<float>(T));
      //(*this)=CSRmatrix(RtensorA::view(const_cast<at::Tensor&>(T)).view2());
    }
    
    at::Tensor torch() const{
      CNINE_CONVERT_TO_ATEN_WARNING();
      Tensor<float> x(*this);
      //RtensorA x(*this);
      return x.torch();
    }

#endif 


  public: // ---- Access -------------------------------------------------------------------------------------


    int nrows() const{
      return n;
    }

    int size_of(const int i) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In CSRmatrix::size_of(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      return array_pool<TYPE>::size_of(i)/2;
    }

    //int lenght_of_row(const int i){
    //CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In CSRmatrix::length_of_row(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
    //return size_of(i)/2;
    //}


    TYPE operator()(const int i, const int j) const{
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In CSRmatrix::operator(): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      CNINE_CHECK_RANGE(if(i>=m) throw std::out_of_range("In CSRmatrix::operator(): index "+to_string(i)+" out of range (0,"+to_string(m-1)+")."));
      CNINE_ASSRT(i<size());
      int offs=dir(i,0);
      int s=dir(i,1);
      for(int a=0; a<s; a++)
	if(*reinterpret_cast<int*>(arr+offs+2*a)==j)
	  return arr[offs+2*a+1];
      CNINE_ASSERT(false,"element("+to_string(i)+","+to_string(j)+") not found");
    }

    void set_at(const int i, const int k, const int j, const TYPE v){
      CNINE_CHECK_RANGE(if(i>=n) throw std::out_of_range("In CSRmatrix::set_at(...): index "+to_string(i)+" out of range (0,"+to_string(n-1)+")."));
      CNINE_CHECK_RANGE(if(k>=size_of(i)) throw std::out_of_range("In CSRmatrix::set_at(...): index "+to_string(k)+" out of range (0,"+to_string(size_of(i)-1)+")."));
      int offs=offset(i);
      *reinterpret_cast<int*>(arr+offs+2*k)=j;
      arr[offs+2*k+1]=v;
    }

    const CSRvector<TYPE> operator()(const int i) const{
      CNINE_CPUONLY();
      CNINE_CHECK_RANGE(if(i>=size()) throw std::out_of_range("In CSRmatrix::operator(): index "+to_string(i)+" out of range (0,"+to_string(size()-1)+")."));
      return CSRvector<TYPE>(arr+dir(i,0),dir(i,1)/2);
    }

    void for_each(std::function<void(const int, const int, const TYPE)> lambda) const{
      if(dev>0){
	CSRmatrix t(*this,0);
	t.for_each(lambda);
	return;
      }
      for(int i=0; i<size(); i++){
	int len=size_of(i);
	int offs=offset(i);
	for(int j=0; j<len; j++)
	  lambda(i,*reinterpret_cast<int*>(arr+offs+2*j),arr[offs+2*j+1]);
      }
    }

    void for_each(std::function<void(const int, const CSRvector<TYPE>)> lambda) const{
      for(int i=0; i<size(); i++)
	lambda(i,(*this)(i));
    }

    void push_back(const vector<int>& ix, const vector<TYPE>& v){
      int len=ix.size();
      CNINE_ASSRT(v.size()==len);
      if(tail+2*len>memsize)
	reserve(std::max(2*memsize,tail+2*len));
      for(int i=0; i<len; i++){
	arr[tail+2*i]=ix[i];
	arr[tail+2*i+1]=v[i];
      }
      dir.push_back(tail,2*len);
      tail+=2*len;
    }


  public: // ---- GPU access ---------------------------------------------------------------------------------

    
    int* get_dirg(const int _dev=1) const{
      return dir.get_arrg(_dev);
    }


  public: // ---- Transposes ---------------------------------------------------------------------------------


    const CSRmatrix& transp() const{
      if(!transpp) make_transp();
      return *transpp;
    }

    void make_transp() const{
      if(transpp) delete transpp;
      transpp=new CSRmatrix<TYPE>(m,n);
      CSRmatrix<TYPE>& T=*transpp;
      T.reserve(tail);

      vector<int> len(m,0);
      for_each([&](const int i, const int j, const TYPE v){
	  len[j]++;});

      T.tail=0;
      for(int i=0; i<m; i++){
	T.dir.set(i,0,T.tail);
	T.dir.set(i,1,2*len[i]);
	T.tail+=2*len[i];
      }

      std::fill(len.begin(),len.end(),0);
      for_each([&](const int i, const int j, const TYPE v){
	  int offs=T.offset(j);
	  *reinterpret_cast<int*>(T.arr+offs+2*len[j])=i;
	  T.arr[offs+2*len[j]+1]=v;
	  len[j]++;
	});
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "CSRmatrix";
    }

    string describe() const{
      return "CSRmarix("+to_string(n)+","+to_string(m)+")";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for_each([&](const int i, const CSRvector<TYPE> lst){oss<<indent<<i<<": "<<lst<<endl;});
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const CSRmatrix& v){
      stream<<v.str(); return stream;}

  };

}


#endif 
