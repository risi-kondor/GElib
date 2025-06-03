#include "Cnine_base.cpp"
#include "CnineSession.hpp"

//#include "for_node.hpp"
//#include "tensor_node.hpp"
//#include "contr_node.hpp"
#include "ptree.hpp"

using namespace cnine;
using namespace etree;



int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  ptree prg;
  auto A=prg.add_input(0,{0,1});
  auto B=prg.add_input(1,{1,2});
  auto C=prg.add_input(2,{2,3});
  auto R=prg.add_output(3,{0,3});
  //prg.add_node(new etree_node());

  /*
  auto U0=prg.add_node(new tensor_node(3,{0,2}));
  auto loop0=prg.add_node(new for_node(0));
  auto loop1=prg.add_node(loop0,new for_node(2));
  auto contr0=prg.add_node(loop1,new contr_node(1,U0,{A,B}));
  prg.add_node(new etree_node());

  //auto R=prg.add_child(new tensor_node(4,{0,3}));
  auto loop2=prg.add_node(new for_node(0));
  auto loop3=prg.add_node(loop2,new for_node(3));
  auto contr1=prg.add_node(loop3,new contr_node(2,R,{U0,C}));

  */
  for(auto& p:prg.nodes->nodes)
    cout<<p->str()<<endl;

  code_env env;
  prg.write_to(env);
  cout<<env.str()<<endl;

}
