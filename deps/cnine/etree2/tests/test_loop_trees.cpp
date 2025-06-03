#include "Cnine_base.cpp"
#include "CnineSession.hpp"

#include "LoopTrees.hpp"

using namespace cnine;


int main(int argc, char** argv){

  cnine_session session;
  cout<<endl;

  ctree ctr;
  auto T0=ctr.add_input({0,1});
  auto T1=ctr.add_input({1,2});
  auto T2=ctr.add_input({2,3});
  auto T3=ctr.add_input({3,4});
  auto T4=contract(T0,T1,1);
  auto T5=contract(T4,T2,2);
  auto T6=contract(T5,T3,3);

  LoopTrees trees(ctr);

  for(auto& p: trees.trees){
    cout<<endl;
    code_env env;
    p->write_to(env);
    cout<<env.str()<<endl;
  }

}

