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


#ifndef __Gdims
#define __Gdims

#include "Cnine_base.hpp"
#include "gvectr.hpp"
#include "GindexSet.hpp"
#include "GindexMap.hpp"
//#include "Bifstream.hpp"
//#include "Bofstream.hpp"


namespace cnine{


  class Gdims: public Gvec<int,Gdims>{
  public:

    typedef Gvec<int,Gdims> BASE;
    typedef std::size_t size_t;

    using BASE::operator[];
    using BASE::push_back;

    using BASE::operator();
    using BASE::operator+;
    using BASE::insert;
    using BASE::remove;
    using BASE::replace;
    using BASE::prepend;
    using BASE::append;
    using BASE::cat;
    using BASE::chunk;
    using BASE::permute;
    using BASE::to_vector;


    Gdims(){}

    Gdims(const vector<int>& x):
      BASE(x){}

    Gdims(const initializer_list<int>& x):
      BASE(x){}

    Gdims(const initializer_list<size_t>& x){
      for(auto& p:x)
	push_back(p);
    }

    explicit Gdims(const int i0): 
      BASE(i0){}

    /*
    Gdims(const int i0, const int i1): 
      BASE(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }
    */

    Gdims(const int k, const fill_raw& dummy): 
      BASE(k){}

    Gdims copy() const{
      return Gdims(*this);
    }


  public: // ---- Named constructors -------------------------------------------------------------------------


    static Gdims raw(const int k){
      return BASE(k);
    }

    static Gdims zero(const int k){
      return BASE(k,0);
    }


  public: // ---- Merging -------------------------------------------------------------------------


    Gdims(const Gdims& d1, const Gdims& d2): 
      BASE(d1.size()+d2.size()){
      for(int i=0; i<d1.size(); i++) (*this)[i]=d1[i];
      for(int i=0; i<d2.size(); i++) (*this)[i+d1.size()]=d2[i];
    }

    Gdims(const int b, const Gdims& d): 
      BASE(d.size()+1){
	(*this)[0]=b;
	std::copy(d.begin(),d.end(),begin()+1);
    }

    /*
    Gdims(const int b, const Gdims& d1, const Gdims& d2): 
      BASE((b>0)+d1.size()+d2.size()){
      if(b>0){
	(*this)[0]=b;
	for(int i=0; i<d1.size(); i++) (*this)[1+i]=d1[i];
	for(int i=0; i<d2.size(); i++) (*this)[1+i+d1.size()]=d2[i];
      }else{
	for(int i=0; i<d1.size(); i++) (*this)[i]=d1[i];
	for(int i=0; i<d2.size(); i++) (*this)[i+d1.size()]=d2[i];
      }
    }
    */

    Gdims(const int b, const Gdims& d1, const Gdims& d2): 
      BASE(1+d1.size()+d2.size()){
      (*this)[0]=b;
      for(int i=0; i<d1.size(); i++) (*this)[1+i]=d1[i];
      for(int i=0; i<d2.size(); i++) (*this)[1+i+d1.size()]=d2[i];
    }

    Gdims(const Gdims& d1, const int v, const Gdims& d2):
      BASE(d1,v,d2){}


  public: // ---- ATEN ---------------------------------------------------------------------------------------


    #ifdef _WITH_ATEN

    Gdims(const at::Tensor& T):
      Gdims(T.dim(),fill_raw()){
      for(int i=0; i<size() ; i++)
	(*this)[i]=T.size(i);
    }

    #endif 

    vector<int64_t> as_int64() const{
      vector<int64_t> v(size());
      for(int i=0; i<size(); i++)
	v[i]=(*this)[i];
      return v;
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    Gdims& set(const int i, const int x){
      BASE::set(i,x);
      return *this;
    }

    Gdims& set_back(const int x){
      BASE::set_back(x);
      return *this;
    }

    Gdims& set_back(const int i, const int x){
      BASE::set_back(i,x);
      return *this;
    }

    size_t asize() const{
      size_t t=1; 
      for(int i=0; i<size(); i++) t*=(*this)[i];
      return t;
    }

    bool valid() const{
      for(auto p:*this)
	if(p<0) return false;
      return true;
    }


  public: // ---- Operations ---------------------------------------------------------------------------------


    int combined(const int a, const int b) const{
      assert(a<=b);
      assert(b<=size());
      int t=1; 
      for(int i=a; i<b; i++) t*=(*this)[i];
      return t;
    }

    Gdims fuse(const int a, const int n) const{
      assert(n<=size());
      assert(a+n<=size());
      Gdims R(size()-n+1,fill_raw());
      for(int i=0; i<a; i++) R[i]=(*this)[i];
      for(int i=0; i<size()-(a+n); i++) R[a+i+1]=(*this)[a+i+n];
      int t=1; for(int i=0; i<n; i++) t*=(*this)[a+i];
      R[a]=t;
      return R;
    }

    Gdims unsqueeze(const int d) const{
      return insert(d,1);
    }

    Gdims transp() const{
      int len=size();
      assert(len>=2);
      if(len==2) return Gdims({(*this)[1],(*this)[0]});
      Gdims r(*this);
      std::swap(r[len-2],r[len-1]);
      return r;
    }

    Gdims transpose() const{
      assert(size()==2);
      return Gdims({(*this)[1],(*this)[0]});
    }
    
    Gdims convolve(const Gdims& y) const{
      assert(size()==y.size());
      Gdims R(*this);
      for(int i=0; i<size(); i++)
	R[i]-=y[i]-1;
      return R;
    }


  public: // In-place operations 


    Gdims& extend(const vector<int>& x){
      for(auto p: x)
	BASE::push_back(p);
      return *this;
    }


  public: // ---- Products -----------------------------------------------------------------------------------


    Gdims Mprod(const Gdims& y) const{
      Gdims R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i+1];
      return R;
    }

    Gdims Mprod_AT(const Gdims& y) const{
      Gdims R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i];
      return R;
    }

   Gdims Mprod_TA(const Gdims& y) const{
      Gdims R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i+1];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i+1];
      return R;
    }

    Gdims Mprod_TT(const Gdims& y) const{
      Gdims R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i+1];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i];
      return R;
    }

    Gdims mvprod(const Gdims& y) const{
      int d=size()-1;
      CNINE_ASSRT(y.size()==d);
      CNINE_ASSRT(chunk(0,d-1)==y.chunk(0,d-1));
      CNINE_ASSRT((*this)[d]==y[d-1]);
      Gdims R(d,fill::raw);
      for(int i=0; i<d-1; i++) R[i]=(*this)[i];
      R[d-1]=(*this)[d-1];
      return R;
    }


  public: // ---- IndexSet -----------------------------------------------------------------------------------


    Gdims select(const GindexSet& s) const{
      Gdims r;
      for(auto p: s){
	assert(p<size());
	r.push_back((*this)[p]);
      }
      return r;
    }

    int unite(const GindexSet& s) const{
      int r=1;
      for(auto p: s){
	assert(p<size());
	r*=(*this)[p];
      }
      return r;
    }


  public: // ---- IndexMap -----------------------------------------------------------------------------------

    
    Gdims map(const GindexMap& map) const{
      CNINE_ASSRT(map.ndims()==size());
      int n=map.size();
      Gdims R=Gdims::zero(n);
      for(int i=0; i<n; i++){
	auto& ix=map[i];
	CNINE_ASSRT(ix.size()>0);
	CNINE_ASSRT(ix[0]<size());
	R[i]=(*this)[ix[0]];
	for(int j=1; j<ix.size(); j++){
	  CNINE_ASSRT(ix[j]<size());
	  CNINE_ASSRT((*this)[ix[j]]==(*this)[ix[0]]);
	}
      }
      return R;
    }


  public: // ---- Strides -----------------------------------------------------------------------------------

    
    vector<int> strides() const{
      int k=size();
      vector<int> R(k);
      if(k==0) return R;
      R[k-1]=1;
      for(int i=k-2; i>=0; i--)
	R[i]=(*this)[i+1]*R[i+1];
      return R;
    }
    

  public: // ---- Lambdas -----------------------------------------------------------------------------------


    void for_each(const std::function<void(const vector<int>&)>& lambda) const{
      foreach_index(lambda);
    }

    void for_each_index(const std::function<void(const vector<int>&)>& lambda) const{
      foreach_index(lambda);
    }

    void foreach_index(const std::function<void(const vector<int>&)>& lambda) const{
      int k=size();
      if(k==0) return;
      vector<int> strides(k);
      strides[k-1]=1;
      for(int i=k-2; i>=0; i--) 
	strides[i]=strides[i+1]*(*this)[i+1];
      int tot=strides[0]*(*this)[0];
      for(int i=0; i<tot; i++){
	vector<int> ix(k);
	int t=i;
	for(int j=0; j<k; j++){
	  ix[j]=t/strides[j];
	  t-=ix[j]*strides[j];
	}
	lambda(ix);
      }
    }


  public: // ---- Checks ------------------------------------------------------------------------------------


    void check_eq(const Gdims& x) const{
      if(!((*this)==x)) throw std::out_of_range("Tensor dimensions "+str()+" do not match "+x.str()+".");
    }

    void check_cell_eq(const Gdims& x) const{
      if(!((*this)==x)) throw std::out_of_range("Tensor cell dimensions "+str()+" do not match "+x.str()+".");
    }

    void check_in_range(const vector<int> ix, const string name) const{
      if(size()!=ix.size()) throw std::out_of_range("cnine::"+name+" index "+Gdims(ix).str()+" out of range of "+str());
      for(int i=0; i<size(); i++)
	if(ix[i]<0 || ix[i]>=(*this)[i]) throw std::out_of_range("cnine::"+name+" index "+Gdims(ix).str()+" out of range of "+str());
    }

    void check_in_range(const int i0, const string name) const{
      CNINE_CHECK_RANGE(if(size()!=1 || i0<0 || i0>=(*this)[0]) 
	  throw std::out_of_range("cnine::"+name+" index "+Gdims({i0}).str()+" out of range of "+str()));
    }

    void check_in_range(const int i0, const int i1, const string name) const{
      CNINE_CHECK_RANGE(if(size()!=2 || i0<0 || i0>=(*this)[0] || i1<0 || i1>=(*this)[1]) 
	  throw std::out_of_range("cnine::"+name+" index "+Gdims({i0,i1}).str()+" out of range of "+str()));
    }

    void check_in_range(const int i0, const int i1, const int i2, const string name) const{
      CNINE_CHECK_RANGE(if(size()!=3 || i0<0 || i0>=(*this)[0] || i1<0 || i1>=(*this)[1] || i2<0 || i2>=(*this)[2]) 
	  throw std::out_of_range("cnine::"+name+" index "+Gdims({i0,i1,i2}).str()+" out of range of "+str()));
    }

    void check_in_range(const int i0, const int i1, const int i2, const int i3, const string name) const{
      CNINE_CHECK_RANGE(if(size()!=4 || i0<0 || i0>=(*this)[0] || i1<0 || i1>=(*this)[1] || i2<0 || i2>=(*this)[2] || i3<0 || i3>=(*this)[3]) 
	  throw std::out_of_range("cnine::"+name+" index "+Gdims({i0,i1,i2,i3}).str()+" out of range of "+str()));
    }

    void check_in_range_d(const int d, const int i0, const string name="") const{
      CNINE_CHECK_RANGE(if(size()<=d || i0<0 || i0>=(*this)[d]) 
	  throw std::out_of_range("cnine::"+name+" index "+Gdims({i0}).str()+" out of range of dimension "+to_string(d)+" of "+str()+"."));
    }


  public: // ---- Deprecated --------------------------------------------------------------------------------

    [[deprecated]]
    Gdims(const int i0, const int i1, const int i2): 
      BASE(3){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
    }

    [[deprecated]]
    Gdims(const int i0, const int i1, const int i2, const int i3): 
      BASE(4){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
    }

    [[deprecated]]
    Gdims(const vector<vector<int> >& list){
      int n=0; 
      for(auto& p:list) n+=p.size();
      resize(n);
      int i=0;
      for(auto& p:list)
	for(auto q:p)
	  (*this)[i++]=q;
    }

    size_t total() const{
      size_t t=1; 
      for(int i=0; i<size(); i++) t*=(*this)[i];
      return t;
    }

    template<typename TYPE>
    vector<TYPE> to_vec() const{
      return BASE::to_vector<TYPE>();
    }


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str() const{
      ostringstream oss;
      int k=size();
      oss<<"(";
      for(int i=0; i<k; i++){
	oss<<(*this)[i];
	if(i<k-1) oss<<",";
      }
      oss<<")";
      return oss.str();
    }

    string repr() const{
      return "<cnine::Gdims"+str()+">";
    }

    friend ostream& operator<<(ostream& stream, const Gdims& x){
      stream<<x.str(); return stream;
    }


  };


  inline Gdims dims(const int i0) {return Gdims({i0});}
  inline Gdims dims(const int i0, const int i1) {return Gdims({i0,i1});}
  inline Gdims dims(const int i0, const int i1, const int i2) {return Gdims({i0,i1,i2});}
  inline Gdims dims(const int i0, const int i1, const int i2, const int i3) {return Gdims({i0,i1,i2,i3});}
  inline Gdims dims(const int i0, const int i1, const int i2, const int i3, const int i4) {return Gdims({i0,i1,i2,i3,i4});}


  template<typename OBJ>
  class as_shape_tmp: public OBJ{
  public:
    as_shape_tmp(const OBJ& x, const Gdims& _dims): OBJ(x,fill::view){
      OBJ::reshape(_dims);}
  };

  inline Gdims tprod(const Gdims& x, const Gdims& y){
    CNINE_ASSRT(x.size()==y.size());
    Gdims r(x.size(),fill_raw());
    for(int i=0; i<x.size(); i++)
      r[i]=x[i]*y[i];
    return r;
  }


}


namespace std{

  template<>
  struct hash<cnine::Gdims>{
  public:
    size_t operator()(const cnine::Gdims& dims) const{
      size_t t=0;
      for(int i=0; i<dims.size(); i++) t=(t^hash<int>()(dims[i]))<<1;
      return t;
    }
  };

}



#endif
    /*
    int operator()(const int i) const{
      if(i<0) return (*this)[size()+i];
      return (*this)[i];
    }

    int back(const int i=0) const{
      return (*this)[size()-1-i];
    }

    Gdims& set(const int i, const int x){
      (*this)[i]=x;
      return *this;
    }

    Gdims& set_back(const int x){
      (*this)[size()-1]=x;
      return *this;
    }

    Gdims& set_back(const int i, const int x){
      (*this)[size()-1-i]=x;
      return *this;
    }

    int first() const{
      return (*this)[0];
    }

    int last() const{
      return (*this)[size()-1];
    }
    */
    /*
    Gdims(const int i0, const int i1, const int i2, const int i3, const int i4): 
      vector<int>(5){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
      (*this)[4]=i4;
    }

    Gdims(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5): 
      vector<int>(6){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
      (*this)[4]=i4;
      (*this)[5]=i5;
    }

    Gdims(const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6): 
      vector<int>(7){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
      (*this)[4]=i4;
      (*this)[5]=i5;
      (*this)[6]=i6;
    }
    */

    /*
    bool operator<=(const Gdims& x) const{
      if(size()!=x.size()) return false;
      for(size_t i=0; i<size(); i++)
	if((*this)[i]>x[i]) return false;
      return true;
    }
    */
    //bool operator==(const Gdims& x) const{
    //if(size()!=x.size()) return false;
    //for(size_t i=0; i<size(); i++)
    //if((*this)[i]!=x[i]) return false;
    //return true;
    //}


    /*
    Gdims append(const int i) const{
      Gdims R=*this;
      if(i>=0) R.push_back(i);
      return R;
    }

    Gdims cat(const Gdims& y) const{
      Gdims R(size()+y.size(),fill_raw());
      for(int i=0; i<size(); i++) R[i]=(*this)[i];
      for(int i=0; i<y.size(); i++) R[size()+i]=y[i];
      return R;
    }

    Gdims prepend(const int i) const{
      if(i<0) return *this;
      Gdims R;
      R.push_back(i);
      for(auto p:*this) R.push_back(p);
      return R;
    }
    */
    /*
    Gdims remove(const int j) const{
      Gdims R;
      assert(j<size());
      if(size()==1){
	R.push_back(1);
	return R;
      }
      if(j<0){
	for(int i=0; i<size(); i++)
	  if(i!=size()+j) R.push_back((*this)[i]);
      }else{
	for(int i=0; i<size(); i++)
	  if(i!=j) R.push_back((*this)[i]);
      }
      return R;
    }

    Gdims insert(const int j, const int n) const{
      Gdims R;
      for(int i=0; i<j; i++) R.push_back((*this)[i]);
      R.push_back(n);
      for(int i=j; i<size(); i++) R.push_back((*this)[i]);
      return R;
    }

    Gdims replace(const int j, const int x) const{
      Gdims R(*this);
      assert(j<size());
      R[j]=x;
      return R;
    }
    */
    /*
    Gdims permute(const vector<int>& p) const{
      CNINE_ASSRT(p.size()<=size());
      Gdims R;
      R.resize(size());
      for(int i=0; i<p.size(); i++)
	R[i]=(*this)[p[i]];
      for(int i=p.size(); i<size(); i++)
	R[i]=p[i];
      return R;
    }
    */
    //Gdims remove(const vector<int>& v) const{
    //return cnine::except(*this,v);
    //}
    //Gdims(const int k, const fill_zero& dummy): 
    //BASE(k,0){}

