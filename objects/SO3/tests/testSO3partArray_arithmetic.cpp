#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CscalarObj Cscalar;
typedef CtensorObj Ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  Gdims adims({2,2});

  SO3partArray u(adims,2,2,fill::gaussian);
  SO3partArray v(adims,2,2,fill::gaussian);
  SO3part w(2,2,fill::gaussian);
  printl("u",u);
  printl("v",v);
  printl("w",w);

  CtensorArray Marr(adims,{2,2},fill::gaussian);
  Ctensor M({2,2},fill::gaussian);
  Ctensor C(adims,fill::gaussian);

  complex<float> cf(2.0);
  Cscalar c(cf);


  // Cellwise arithmetic 

  printl("u+v",u+v);
  printl("u-v",u-v);

  printl("Marr*u",Marr*u);
  printl("transp(Marr)*u",transp(Marr)*u);


  // Broadcasting arithmetic

  printl("c*u",c*u);
  printl("u*c",u*c);
  printl("cf*u",cf*u);
  printl("u*cf",u*cf);

  printl("u/c",u/c);
  printl("u/cf",u/cf);

  printl("u+w",u+w);
  printl("w+u",w+u);

  printl("u-w",u-w);
  printl("w-u",w-u);

  printl("M*u",M*u);
  printl("transp(M)*u",transp(M)*u);

  // Scattering arithmetic

  printl("scatter(C)*u",scatter(C)*u);
  printl("u*scatter(C)",u*scatter(C));
  printl("u/scatter(C)",u/scatter(C));


  // Inplace cellwise arithmetic 

  printl("u+=v",u+=v);
  printl("u-=v",u-=v);


  // Inplace broadcast arithmetic 

  printl("u*=c",u*=c);
  printl("u*=cf",u*=cf);

  printl("u/=c",u/=c);
  printl("u/=cf",u/=cf);


  // Inplace scattering arithmetic

  printl("u*=scatter(C)",u*=scatter(C));
  printl("u/=scatter(C)",u/scatter(C));

}


