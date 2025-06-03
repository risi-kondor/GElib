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


#ifndef _object_pack_s
#define _object_pack_s

#include "Cnine_base.hpp"

namespace cnine{

  template<typename OBJ>
  class object_pack_s{
  public:


    vector<shared_ptr<OBJ> > obj;


  public: // ---- Constructors -------------------------------------------------------------------------------


    object_pack_s(){}

    object_pack_s(const vector<shared_ptr<OBJ> >& x): 
      obj(x){}


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return obj.size();
    }

    const OBJ& operator[](const int i) const{
      CNINE_ASSRT(i<obj.size());
      CNINE_ASSRT(obj[i].get()!=nullptr);
      return *obj[i];
    }

    OBJ& operator[](const int i){
      CNINE_ASSRT(i<obj.size());
      CNINE_ASSRT(obj[i].get()!=nullptr);
      return *obj[i];
    }

    void push_back(const shared_ptr<OBJ>& x){
      obj.push_back(x);
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    template<typename OTHER>
    void zip(const OTHER& x, const std::function<void(OBJ&, const OBJ&)>& fn){
      CNINE_ASSRT(obj.size()==x.size());
      for(int i=0; i<obj.size(); i++)
	fn((*this)[i],x[i]);
    }
    
    template<typename PACK, typename OBJ2>
    PACK mapcar(const std::function<OBJ2(const OBJ&)>& fn){
      PACK R;
      for(auto& p:obj)
	R.obj.push_back(to_share(new OBJ2(fn(*p))));
      return R;
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    static string classname(){
      return "object_pack_s<"+OBJ::classname()+">";
    }

    string repr() const{
      return "<object_pack_s<"+OBJ::classname()+">(N="+to_string(size())+">";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<obj.size(); i++){
	oss<<"Object "<<i<<":\n";
	oss<<(*this)[i].str(indent+"  ");
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const object_pack_s& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
