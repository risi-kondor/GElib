//#define DEBUG_ENGINE_FLAG
//#define ENGINE_PRIORITY 

#include "GElib_base.cpp"
#include "SO3part.hpp"
#include "SO3vec.hpp"
#include "GElibSession.hpp"

using namespace cnine;
using namespace GElib;

typedef CscalarObj cscalar;
typedef CtensorPackObj ctensorpack;


int main(int argc, char** argv){
  GElibSession session;
  cout<<endl;

  SO3type tau({2,2,2});

  SO3vec u(tau,fill::gaussian,SO3vec_format::compact);
  SO3vec v(tau,fill::gaussian,SO3vec_format::compact);
  cout<<"u="<<endl<<u<<endl<<endl;
  cout<<"v="<<endl<<v<<endl<<endl;

  cout<<"u+u+v="<<endl<<u+u+v<<endl<<endl;
  cout<<"u-v="<<endl<<u-v<<endl<<endl;

  cscalar c=2.0;
  cout<<"c*u="<<endl<<c*u<<endl<<endl; 

  u+=v;
  cout<<"u="<<endl<<u<<endl<<endl;

  //u.set(1,0,1,3.0);
  //printl("u",u);
  //cout<<"u(1,0,0)="<<u(1,0,0)<<endl;

  //printl("cat(u,u)=",cat(u,u));
  //cout<<endl;
  
  SO3part P=u.get_part(1);
  cout<<P<<endl<<endl;
  v.set_part(1,P);
  printl("v",v);


  cout<<"-------"<<endl;

  SO3vec a=u-v;
  cout<<a<<endl;

  u=u*100;
  
  printl("u",u);


  cout<<"norm2(u) = "<<norm2(u)<<endl<<endl; 
  cout<<"inp(u,v)= "<<endl<<inp(u,v)<<endl<<endl; 

  cout<<"CGproduct(u,v)="<<endl<<CGproduct(u,v)<<endl;

  //u.normalize_fragments();
  //cout<<"u.normalize_fragments()="<<endl<<u<<endl;

  //u=v;
  //cout<<u<<endl;

  cout<<endl; 

  ctensorpack W(dims(tau,tau),fill::gaussian);
  cout<<W*u<<endl;

}


  /*
  if(false){
    SO3vecObj u(tau,fill::gaussian,SO3vec_format::joint);
    SO3vecObj v(tau,fill::gaussian,SO3vec_format::joint);
    cout<<"u="<<endl<<u<<endl<<endl;
    cout<<"v="<<endl<<v<<endl<<endl;

    cout<<"u+v="<<endl<<u+v<<endl<<endl;

    //cout<<"inp(u,v)= "<<endl<<inp(u,v)<<endl<<endl; 
    
    //cout<<"CGproduct(u,v)="<<endl<<CGproduct(u,v)<<endl;


    cout<<"........................."<<endl;
  }
  */
