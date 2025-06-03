#ifndef CnineSharedCref
#define CnineSharedCref

namespace cnine{

  template<typename OBJ>
  class shared_cref{
  public:

    shared_ptr<const OBJ> p;

    shared_cref(){}

    shared_cref(const OBJ& _p): p(&_p){}

    operator const OBJ&() const{
      return *p;
    }
    
    const OBJ& obj() const{
      return *p;
    }

  };


}

#endif 
