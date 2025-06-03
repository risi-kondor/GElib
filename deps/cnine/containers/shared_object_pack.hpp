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


#ifndef _shared_object_pack
#define _shared_object_pack

#include "Cnine_base.hpp"

namespace cnine{


  template<typename OBJ>
  class shared_object_pack: public vector<shared_ptr<OBJ> >{
  public:

    typedef vector<shared_ptr<OBJ> > BASE;

    //vector<shared_ptr<OBJ> > obj;

    using BASE::size;


  public: // ---- Constructors -------------------------------------------------------------------------------


    shared_object_pack(){}

    shared_object_pack(const vector<shared_ptr<OBJ> >& x): 
      BASE(x){}


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return BASE::size();
    }

    shared_ptr<OBJ> operator()(const int i) const{
      CNINE_ASSRT(i<BASE::size());
      return BASE::operator[](i);
    }      

    const OBJ& operator[](const int i) const{
      CNINE_ASSRT(i<size());
      CNINE_ASSRT(BASE::operator[](i).get()!=nullptr);
      return *BASE::operator[](i);
    }

    OBJ& operator[](const int i){
      CNINE_ASSRT(i<BASE::size());
      CNINE_ASSRT(BASE::operator[](i).get()!=nullptr);
      return *BASE::operator[](i);
    }

    void push_back(const OBJ& x){
      BASE::push_back(shared_ptr<OBJ>(new OBJ(x)));
    }

    void push_back(const shared_ptr<OBJ>& x){
      BASE::push_back(x);
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    template<typename OTHER>
    void zip(const OTHER& x, const std::function<void(OBJ&, const OBJ&)>& fn){
      CNINE_ASSRT(size()==x.size());
      for(int i=0; i<size(); i++)
	fn((*this)[i],x[i]);
    }
    
    template<typename PACK, typename OBJ2>
    PACK mapcar(const std::function<OBJ2(const OBJ&)>& fn){
      PACK R;
      for(auto& p:*this)
	R.push_back(to_share(new OBJ2(fn(*p))));
      return R;
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    static string classname(){
      return "shared_object_pack<"+OBJ::classname()+">";
    }

    string repr() const{
      return "shared_object_pack";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<size(); i++){
	oss<<"Object "<<i<<":\n";
	oss<<(*this)[i].str(indent+"  ");
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const shared_object_pack& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
