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


#ifndef __GdimsPack
#define __GdimsPack

#include "Cnine_base.hpp"
#include "Gdims.hpp"
//#include "Bifstream.hpp"
//#include "Bofstream.hpp"


namespace cnine{


  class GdimsPack: public vector<Gdims>{
  public:

    GdimsPack(){}

    //GdimsPack(const vector<int>& x): vector<int>(x){}

    //GdimsPack(const int k, const fill_raw& dummy): 
    //vector<int>(k){}

    //GdimsPack(const vector<int>& x){
    //for(auto p:x) if(p>=0) push_back(p);
    //}

    GdimsPack(const Gdims& i0): vector<Gdims>(1){
      (*this)[0]=i0;
    }

    GdimsPack(const Gdims& i0, const Gdims& i1): vector<Gdims>(2){
      (*this)[0]=i0;
      (*this)[1]=i1;
    }

    GdimsPack(const Gdims& i0, const Gdims& i1, const Gdims& i2): vector<Gdims>(3){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
    }

    GdimsPack(const Gdims& i0, const Gdims& i1, const Gdims& i2, const Gdims& i3): vector<Gdims>(4){
      (*this)[0]=i0;
      (*this)[1]=i1;
      (*this)[2]=i2;
      (*this)[3]=i3;
    }


    GdimsPack(const initializer_list<initializer_list<int> >& x){
      for(auto& p:x) 
	push_back(Gdims(p));
    }


  public:

    //int k() const{
    //return size();
    //}

    Gdims operator()(const int i) const{
      return (*this)[i];
    }

    //int asize() const{
    //int t=1; for(int i=0; i<size(); i++) t*=(*this)[i];
    //return t;
    //}

    Gdims first() const{
      return (*this)[0];
    }

    Gdims last() const{
      return (*this)[size()-1];
    }



    /*
    GdimsPack transpose() const{
      assert(size()==2);
      return GdimsPack((*this)[1],(*this)[0]);
    }

    GdimsPack Mprod(const GdimsPack& y) const{
      GdimsPack R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i+1];
      return R;
    }

    GdimsPack Mprod_AT(const GdimsPack& y) const{
      GdimsPack R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i];
      return R;
    }

   GdimsPack Mprod_TA(const GdimsPack& y) const{
      GdimsPack R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i+1];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i+1];
      return R;
    }

   GdimsPack Mprod_TT(const GdimsPack& y) const{
      GdimsPack R(size()+y.size()-2,fill::raw);
      for(int i=0; i<size()-1; i++) R[i]=(*this)[i+1];
      for(int i=0; i<y.size()-1; i++) R[i+size()-1]=y[i];
      return R;
   }
    */


  public:

    /*
    GdimsPack(Bifstream& ifs){
      int _k=ifs.get<int>();
      resize(_k);
      for(int i=0; i<_k; i++)
	(*this)[i]=ifs.get<int>();
    }

    void serialize(Bofstream& ofs) const{
      const int k=size();
      ofs.write(k);
      for(int i=0; i<k; i++)
	ofs.write((*this)[i]);
    }
    */

    string str() const{
      ostringstream oss;
      /*
      int k=size();
      oss<<"(";
      for(int i=0; i<k; i++){
	oss<<(*this)[i];
	if(i<k-1) oss<<",";
        }
      oss<<")";
      */
     return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const GdimsPack& x){
      stream<<x.str(); return stream;
    }


  };

}


/*
namespace std{

  template<>
  struct hash<cnine::GdimsPack>{
  public:
    size_t operator()(const cnine::GdimsPack& dims) const{
      size_t t=0;
      for(int i=0; i<dims.size(); i++) t=(t^hash<int>()(dims[i]))<<1;
      return t;
    }
  };

}
*/



#endif
