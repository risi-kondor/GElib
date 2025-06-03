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


#ifndef _CnineHmatrix
#define _CnineHmatrix

#include "Cnine_base.hpp"
#include <unordered_map>

#include "Hvector.hpp"
#include "Gdims.hpp"
#include "IntTensor.hpp"
#include "RtensorA.hpp"
#include "CSRmatrix.hpp"


namespace cnine{

  // deprecated 
  template<typename TYPE>
  class Hmatrix{
  public:

    //typedef RtensorA rtensor;

    typedef Hvector<TYPE> Hvector;


    int n=0;
    int m=0;
    unordered_map<int,Hvector> rows;



  public: // ---- Constructors -------------------------------------------------------------------------------


    Hmatrix(){} 

    Hmatrix(const int _n, const int _m): 
      n(_n), m(_m){} 
    

  public: // ---- Copying ------------------------------------------------------------------------------------


    Hmatrix(const Hmatrix& x){
      CNINE_COPY_WARNING();
      n=x.n; 
      m=x.m;
      for(auto p:x.rows) 
	rows[p.first]=p.second; //new Hvector(*p.second);
    }

    Hmatrix(Hmatrix&& x){
      CNINE_MOVE_WARNING();
      n=x.n; 
      m=x.m;
      rows=std::move(x.rows);
      x.rows.clear();
    }

    Hmatrix& operator=(const Hmatrix& x)=delete;


  public: // ---- Named constructors -------------------------------------------------------------------------


    static Hmatrix bernoulli(const int _n, const int _m, const float p=0.5){
      Hmatrix G(_n,_m);
      uniform_real_distribution<double> distr(0,1);
      for(int i=0; i<_n; i++) 
	for(int j=0; j<_m; j++)
	  if(distr(rndGen)<p)
	    G.set(i,j,1.0);
      return G;
    }

    /*
    static Hmatrix from_list(const IntTensor& M){
      CNINE_ASSRT(M.ndims()==2);
      CNINE_ASSRT(M.dim(1)==2);
      int n=0; int m=0;
      for(int i=0; i<M.dim(0); i++){
	n=std::max(M(i,0),n);
	m=std::max(M(i,1),m);
      }
      Hmatrix R(n,m); 
      for(int i=0; i<M.dim(0); i++)
	R.set(M(i,0),M(i,1),1);
      return R;
    }
    */

    /*
    static Hmatrix from_matrix(const IntTensor& A){
      CNINE_ASSRT(A.ndims()==2);
      int n=A.dim(0);
      int m=A.dim(1);
      Hmatrix R(n,m); 
      for(int i=0; i<n; i++)
	for(int j=0; j<m; j++)
	  if(A(i,j)>0) R.set(i,j,A(i,j));
      return R;
    }
    */


  public: // ---- Conversions ------------------------------------------------------------------------------

    /*
    Hmatrix(const rtensor& x){
      CNINE_ASSRT(x.ndims()==2);
      n=x.dim(0);
      m=x.dim(1);
      for(int i=0; i<n; i++)
	for(int j=0; j<m; j++)
	  if(x(i,j)!=0) set(i,j,x(i,j));
    }

    rtensor dense() const{
      auto R=rtensor::zero({n,m});
      forall_nonzero([&](const int i, const int j, const float v){
	  R.set(i,j,v);});
      return R;
    }
    */

  public: // ---- Boolean ----------------------------------------------------------------------------------


    bool operator==(const Hmatrix& x) const{
      for(auto& p: rows){
	auto it=x.rows.find(p.first);
	if(it==x.rows.end()) return false;
	if(p.second!=it->second) return false;
      }
      return true;
    }


  public: // ---- Access -----------------------------------------------------------------------------------


    int getn() const{
      return n;
    }

    int getm() const{
      return m;
    }

    float operator()(const int i, const int j) const{
      auto it=rows.find(i);
      if(it==rows.end()) return TYPE();
      auto it2=it->second.find(j);
      if(it2==it->second.end()) return TYPE();
      return it2->second;
    }
 
    void set(const int i, const int j, const TYPE& v){
      CNINE_ASSRT(i<n);
      CNINE_ASSRT(j<m);
      row(i).set(j,v);
      /*
      auto it=rows.find(i);
      if(it==rows.end()){
	rows.insert({i,Hvector(m)});
	rows[i].set(j,v);
      }else{
	it->second.set(j,v);
      }
      */
    }

    Hvector& row(const int i){
      CNINE_ASSRT(i<n);
      auto it=rows.find(i);
      if(it!=rows.end()) return it->second;
      auto it2=rows.insert({i,Hvector(m)});
      return it2.first->second;
    }

    /*
    const Hvector& row(const int i) const{
      CNINE_ASSRT(i<n);
      if(rows.find(i)==rows.end())
	const_cast<Hmatrix*>(this)->rows[i]=new Hvector(m);
      return *(const_cast<Hmatrix*>(this)->rows[i]);
    }
    */


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    void for_each_nonzero(std::function<void(const int, const int, const TYPE&)> lambda) const{
      for(auto& p: rows){
	int i=p.first;
	p.second.forall_nonzero([&](const int j, const TYPE& v){
	    lambda(i,j,v);});
      }
    }


  public: // ---- Operations -------------------------------------------------------------------------------


    Hmatrix transp() const{
      Hmatrix R(m,n);
      forall_nonzero([&](const int i, const int j, const float v){R.set(j,i,v);});
      return R;
    }


    /*
    CSRmatrix<float> csrmatrix() const{
      cnine::CSRmatrix<float> R;
      R.dir.resize0(n);
      //cout<<R.dir<<endl;
      int t=0;
      for(auto q:rows) t+=2*(q.second->size());
      R.reserve(t);
      
      t=0;
      for(int i=0; i<n; i++){
	R.dir.set(i,0,t);
	auto it=rows.find(i);
	if(it==rows.end()){
	  R.dir.set(i,1,0);
	  continue;
	}
	const Hvector& v=*it->second;
	R.dir.set(i,1,2*v.size());
	for(auto p:v){
	  *reinterpret_cast<int*>(R.arr+t)=p.first;
	  R.arr[t+1]=p.second;
	  t+=2;
	}
      }
      R.tail=t;
      return R;
    }
    */


    public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Hmatrix";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for_each_nonzero([&](const int i, const int j, const TYPE& x){
	  oss<<"("<<i<<","<<j<<")->"<<x<<endl;});
      //for(auto it: rows){
      //oss<<indent<<it.first<<"<-(";
      //oss<<it.second;
      //oss<<")"<<endl;
      //}
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const Hmatrix& x){
      stream<<x.str(); return stream;}
    
  };

}


/*
namespace std{
  template<>
  struct hash<cnine::Hmatrix>{
  public:
    size_t operator()(const cnine::Hmatrix& x) const{
      size_t t=1;
      for(auto& p: x.rows){
	t=(t<<1)^hash<int>()(p.first);
	t=(t<<1)^hash<cnine::Hvector>()(p.second);
      }
      return t;
    }
  };
}
*/


#endif 
