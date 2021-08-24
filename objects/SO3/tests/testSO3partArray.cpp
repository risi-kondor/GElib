#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  Gdims adims({2,2});

  SO3partArray u(adims,2,2,fill::gaussian);
  SO3partArray v(adims,2,2,fill::gaussian);
  SO3partArray w(dims(2),2,2,fill::gaussian);
  SO3partArray z(dims(3,3),2,2,fill::gaussian);
  SO3part a(2,2,fill::gaussian);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  printl("u+u+v",u+u+v);
  printl("u-v",u-v);

  cscalar c=2.0;
  printl("c*u",c*u); 

  SO3part p=u.get_cell({0,0});
  printl("cell(0,0)",p);

  ctensor M({2,2},fill::identity);
  cout<<"M="<<endl<<M<<endl<<endl;

  cout<<"M*u="<<endl<<M*u<<endl<<endl;

  cout<<"------"<<endl;

  printl("CGproduct(u,v,2)",CGproduct(u,v,2));

  //printl("CGproduct(outer(w,w),2)",CGproduct(outer(w,w),2));

  //printl("CGproduct(convolve(z,u),2)",CGproduct(convolve(z,u),2));

  //printl("CGproduct(u,a,2)",CGproduct(u,a,2));

  //printl("CGproduct(a,u,2)",CGproduct(a,u,2));

  cout<<endl; 
}


  /*
  cout<<"-------"<<endl;


  u+=v;
  cout<<"u="<<endl<<u<<endl<<endl;

  cout<<"norm2(u) = "<<norm2(u)<<endl<<endl; 
  cout<<"inp(u,v)= "<<endl<<inp(u,v)<<endl<<endl; 

  */

  //u.normalize_fragments();
  //cout<<"u.normalize_fragments()="<<endl<<u<<endl<<endl;

  //cout<<"u.column_norms="<<u.column_norms()<<endl; 

  //cout<<"-------"<<endl;

  /*
  cout<<u.fragment(0)<<endl; 
  cout<<u.fragment(1)<<endl; 

  cout<<u<<endl; 
  u.fragment(0)=10*u.fragment(1);
  cout<<u<<endl; 


  cout<<SO3part(2,3,[](const int i, const int m){return complex<float>(m,i);})<<endl;
  */
