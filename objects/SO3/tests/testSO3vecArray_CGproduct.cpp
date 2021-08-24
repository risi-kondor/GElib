#include "GElib_base.cpp"
#include "SO3vecArray.hpp"
#include "GElibSession.hpp"

#include "CGprod.hpp"

using namespace cnine;
using namespace GElib;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  Gdims adims({2,2});
  SO3type tau({1,1});

  SO3vecArray u(adims,tau,fill::gaussian);
  SO3vecArray v(adims,tau,fill::gaussian);
  SO3vecArray w(dims(2),tau,fill::gaussian);
  SO3vecArray z(dims(3,3),tau,fill::gaussian);
  SO3vec a(tau,fill::gaussian);

  printl("CGproduct(u,v,2)",CGproduct(u,v,2));
  printl("CGproduct(outer(w,w),2)",CGproduct(outer(w,w),2));
  printl("CGproduct(convolve(z,u),2)",CGproduct(convolve(z,u),2));
  printl("CGproduct(u,a,2)",CGproduct(u,a,2));
  printl("CGproduct(a,u,2)",CGproduct(a,u,2));

  printl("outerprod<CGprod>(w,w,2)",outerprod<CGprod>(w,w,2));
  printl("convolution<CGprod>(z,u,2)",convolution<CGprod>(z,u,2));
  printl("matrixprod<CGprod>(w,w,2)",matrixprod<CGprod>(w,w,2));

  cout<<endl; 
}
