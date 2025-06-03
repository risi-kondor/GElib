#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "code_blocks.hpp"
#include "ctensor_node.hpp"


using namespace cnine;
using namespace etree;



int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  auto A=shared_tnode(0,{0,1});
  auto B=shared_tnode(1,{1,2});
  auto C=shared_tnode(2,{2,3});

  auto U=shared_cnode(3,1,{0,2},{A,B});
  auto R=shared_cnode(4,2,{0,3},{U,C});

  code_env code;
  R->cpu_code(code);
  cout<<code.str()<<endl;

}
