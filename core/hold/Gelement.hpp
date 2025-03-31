#ifndef _GElibGelement
#define _GElibGelement

#include "GElib_base.hpp"
#include "GelementObj.hpp"


namespace GElib{


  class Gelement{
  public:

    const shared_ptr<GelementObj> obj;


    Gelement(GelementObj* x):
      obj(x){}

    Gelement(const shared_ptr<GelementObj>& x):
      obj(x){}

    virtual ~Gelement(){}


  public: // ---- Access ---------------------------------------------------------------------------------


    //GgroupSub group() const{
    //return obj->G;
    //}


  public: // ---- Operations ---------------------------------------------------------------------------------

    
    Gelement inv() const{
      return obj->inv();
    }

    Gelement operator*(const Gelement& y) const{
      return obj->mult(*y.obj);
    }

    Gelement operator^(const int m) const{
      return obj->pow(m);
    }

    bool operator==(const Gelement& y) const{
      return (*obj)==(*y.obj);
    }


  public: // ---- I/O ----------------------------------------------------------------------------------------


    string classname() const{
      return "Gelement";
    }

    string repr() const{
      return obj->repr();
    }
    
    string str(const string indent="") const{
      return obj->str(indent);
    }

    friend ostream& operator<<(ostream& stream, const Gelement& x){
      stream<<x.str(); return stream;
    }


  };

}

#endif 
