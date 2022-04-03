#ifndef _SO3CGproductSignature
#define _SO3CGproductSignature

#include "SO3type.hpp"
//#include "SO3vectorB.hpp"
//#include "SO3vectorVar.hpp"
//#include "SO3vectorVarB.hpp"
//#include "SO3vec.hpp"


namespace GElib{


  class SO3CGproductSignature: public vector<SO3type>{
  public:

    int maxl=-1;

  public:

    SO3CGproductSignature(){}

    SO3CGproductSignature(const initializer_list<SO3type> _taus, const int _maxl=-1): 
      vector<SO3type>(_taus), maxl(_maxl){}

    SO3CGproductSignature(const vector<SO3type>& _taus, const int _maxl=-1): 
      vector<SO3type>(_taus), maxl(_maxl){}

    /*
    template<class OBJ>
    SO3CGproductSignature(const vector<const OBJ*> vecs, const int _maxl=-1): maxl(_maxl){
      resize(vecs.size()); for(int i=0; i<vecs.size(); i++) (*this)[i]=vecs[i]->type();}

    template<class OBJ>
    SO3CGproductSignature(const vector<OBJ*> vecs, const int _maxl=-1): maxl(_maxl){
      resize(vecs.size()); for(int i=0; i<vecs.size(); i++) (*this)[i]=vecs[i]->type();}

    template<class TYPE>
    SO3CGproductSignature(const vector<const SO3vector<TYPE>*> vecs, const int _maxl=-1): maxl(_maxl){
      resize(vecs.size()); for(int i=0; i<vecs.size(); i++) (*this)[i]=vecs[i]->type();}

    template<class TYPE>
    SO3CGproductSignature(const vector<const SO3vectorB<TYPE>*> vecs, const int _maxl=-1): maxl(_maxl){
      resize(vecs.size()); for(int i=0; i<vecs.size(); i++) (*this)[i]=vecs[i]->type();}

    template<class TYPE>
    SO3CGproductSignature(const vector<SO3vectorVar<TYPE>*> vecs, const int _maxl=-1): maxl(_maxl){
      resize(vecs.size()); for(int i=0; i<vecs.size(); i++) (*this)[i]=vecs[i]->type();}

    template<class TYPE>
    SO3CGproductSignature(const vector<SO3vectorVarB<TYPE>*> vecs, const int _maxl=-1): maxl(_maxl){
      resize(vecs.size()); for(int i=0; i<vecs.size(); i++) (*this)[i]=vecs[i]->type();}
    */

    //  template<class TYPE>
    //SO3CGproductSignature(const vector<const SO3vec<TYPE>*>& vecs, const int _maxl=-1): maxl(_maxl){
    //  resize(vecs.size()); for(int i=0; i<vecs.size(); i++) (*this)[i]=vecs[i]->type();}


  public:

    string str() const{
      ostringstream oss;
      oss<<"[ ";
      for(auto& p:*this) oss<<p<<" ";
      oss<<"]";
      if(maxl>-1) oss<<"(maxl="<<maxl<<")";
      return oss.str();
    }

  };

}


namespace std{
  template<>
  struct hash<GElib::SO3CGproductSignature>{
  public:
    size_t operator()(const GElib::SO3CGproductSignature& T) const{
      size_t h=hash<int>()(T.maxl);
      for(auto& p:T)
	h=(h<<1)^hash<GElib::SO3type>()(p);
      return h;
    }
  };
}


#endif

