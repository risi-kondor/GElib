#include "GElib_base.cpp"
#include "SO3part.hpp"
#include "GElibSession.hpp"


using namespace cnine;
using namespace GElib;

typedef CscalarObj cscalar;
typedef CtensorObj ctensor;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;


  SO3part u(2,2,fill::gaussian);
  SO3part v(2,2,fill::gaussian);
  printl("u",u)<<endl;
  printl("v",v)<<endl;

  cout<<"u+u+v="<<endl<<u+u+v<<endl<<endl;
  cout<<"u-v="<<endl<<u-v<<endl<<endl;

  cscalar c=2.0;
  cout<<"c*u="<<endl<<c*u<<endl<<endl; 

  u.set_value(0,1,3.0);
  printl("u",u);
  cout<<endl;

  cout<<"u(0,0)="<<u(0,0)<<endl<<endl;

  printl("u.chunk(0)",u.chunk(0));
  cout<<endl;

  //printl("cat(u,u)=",cat(u,u));

  cout<<"-------"<<endl;

  ctensor M({2,2},fill::identity);
  cout<<"M="<<endl<<M<<endl<<endl;

  cout<<"M*u="<<endl<<M*u<<endl<<endl;

  u+=v;
  cout<<"u="<<endl<<u<<endl<<endl;

  cout<<"norm2(u) = "<<norm2(u)<<endl<<endl; 
  cout<<"inp(u,v)= "<<endl<<inp(u,v)<<endl<<endl; 

  cout<<"CGproduct(u,v,2)="<<endl<<CGproduct(u,v,2)<<endl<<endl;;

  cout<<"DiagCGproduct(u,v,2)="<<endl<<DiagCGproduct(u,v,2)<<endl<<endl;;

  //u.normalize_fragments();
  //cout<<"u.normalize_fragments()="<<endl<<u<<endl<<endl;

  //cout<<"u.column_norms="<<u.column_norms()<<endl; 

  cout<<"-------"<<endl;

  /*
  cout<<u.fragment(0)<<endl; 
  cout<<u.fragment(1)<<endl; 

  cout<<u<<endl; 
  u.fragment(0)=10*u.fragment(1);
  cout<<u<<endl; 


  cout<<SO3part(2,3,[](const int i, const int m){return complex<float>(m,i);})<<endl;
  */

  cout<<endl; 
}
