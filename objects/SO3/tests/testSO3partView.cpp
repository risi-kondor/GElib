#define ENGINE_PRIORITY 

#include "GEnet_base.cpp"
#include "SO3vecVar_funs.hpp"
#include "Dfunction2.hpp"

using namespace GEnet;


int main(int argc, char** argv){

  GEnetSession genet;

  cout<<endl;

  SO3type tau({1,1,1});

  SO3vecVar u(tau,fill::gaussian);
  SO3vecVar v(tau,fill::gaussian);
  printl("u",u);
  printl("v",v);


  SO3partVar p=u[1];
  print(p); 
  cout<<endl; 

  /*
  cout<<GEnet_global_program<<endl; 

  u[1]+=v[1];
  print(u);

  u.set_gradient(SO3vec(tau,fill::ones));
  u.backward();

  print(v.grad());
  */


  Dfunction2<SO3vecVar,SO3vec>
    foo([](const SO3vec _a){
	SO3vecVar a(_a);
	SO3vecVar b({1,1,1},fill::gaussian);
	SO3vecVar c=a+a; 
	printl("a",a);
	printl("b",b);
	a[1]+=b[1];
	return a;
      });

  SO3vec _a({1,1,1},fill::gaussian);
  printl("a",foo(_a));
  
  SO3vec g(tau,fill::ones);
  printl("g",g);
  //foo.backward(g);
  foo.backward(SO3vec(tau,fill::ones));

  cout<<foo.local_prog<<endl; 
  cout<<foo.local_prog.print_backward()<<endl; 

}
