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


#ifndef _object_pack
#define _object_pack

#include "Cnine_base.hpp"

namespace cnine{

  template<typename OBJ>
  class object_pack{
  public:


    vector<OBJ> obj;


  public: // ---- Constructors -------------------------------------------------------------------------------


    object_pack(){}

    object_pack(const initializer_list<OBJ>& list){
      for(auto& p:list)
	obj.push_back(p);
    }


  public: // ---- Access -------------------------------------------------------------------------------------


    int size() const{
      return obj.size();
    }

    OBJ& operator[](const int i){
      CNINE_ASSRT(i<obj.size());
      return obj[i];
    }

    const OBJ& operator[](const int i) const{
      CNINE_ASSRT(i<obj.size());
      return obj[i];
    }


  public: // ---- Lambdas ------------------------------------------------------------------------------------


    template<typename OTHER>
    void zip(const OTHER& x, const std::function<void(const OBJ&, const OBJ&)>& fn){
      CNINE_ASSRT(obj.size()==x.size());
      for(int i=0; i<obj.size(); i++)
	fn((*this)[i],x[i]);
    }
    
    template<typename PACK, typename OBJ2>
    PACK mapcar(const std::function<OBJ2(const OBJ&)>& fn){
      PACK R;
      for(auto& p:obj)
	R.obj.push_back(fn(p));
      return R;
    }


  public: // ---- I/O -----------------------------------------------------------------------------------------


    static string classname(){
      return "object_pack<"+OBJ::classname()+">";
    }

    string repr() const{
      return "object_pack";
    }

    string str(const string indent="") const{
      ostringstream oss;
      for(int i=0; i<obj.size(); i++){
	oss<<"Object "<<i<<":\n";
	oss<<(*this)[i].str(indent+"  ");
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const object_pack& x){
      stream<<x.str(); return stream;}


  };

}

#endif 
