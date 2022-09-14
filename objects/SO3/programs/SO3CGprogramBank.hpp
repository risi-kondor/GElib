#ifndef _SO3CGprogramBank
#define _SO3CGprogramBank

#include "SO3type.hpp"
#include "SO3CGprogram.hpp"
#include "SO3vecB.hpp"
#include "SO3CGproductSignature.hpp"

#include "SO3CGP_addCGproductPrg.hpp"
    

namespace GElib{


  class SO3CGprogramBank{
  public:

    unordered_map<SO3CGproductSignature,SO3CGprogram*> programs; 

    SO3CGprogramBank(){}

    SO3CGprogramBank(const SO3CGprogramBank& x)=delete;
    SO3CGprogramBank& operator=(const SO3CGprogramBank& x)=delete;

    ~SO3CGprogramBank(){
      for(auto& p: programs) delete p.second;
    }

    string classname() const {return "FastCG::SO3CGprogramBank";}


  public:

    const SO3CGprogram& add_CGproduct(const vector<SO3type>& v, const int maxl=-1){
      return add_CGproduct(SO3CGproductSignature(v,maxl));
    }

    const SO3CGprogram& add_CGproduct(const SO3CGproductSignature& taus){
      auto it=programs.find(taus);
      if(it!=programs.end()) return *it->second;
      cout<<"Generating new program for "<<taus.str()<<endl;
      programs[taus]=new SO3CGP_addCGproductPrg(taus);
      return *programs[taus];
    }




  };
  
}



#endif
