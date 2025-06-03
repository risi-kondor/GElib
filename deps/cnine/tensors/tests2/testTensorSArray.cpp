#include "Cnine_base.cpp"
#include "CnineSession.hpp"
#include "Tensor.hpp"
#include "TensorSArray.hpp"
#include "map_of_lists.hpp"


using namespace cnine;

int main(int argc, char** argv){
  cnine_session session;
  cout<<endl;

  Gdims adims({3,3});
  Gdims ddims({2,2});

  map_of_lists<int,int> mol({{0,{1,2}},{2,{0}}});
  cout<<mol<<endl;

  //SparseTensor<int> mask=SparseTensor<int>::ones(mol)


  TensorSArray<float> u=TensorSArray<float>::sequential(adims,ddims,mol);
  printl("u",u)<<endl;

  cout<<endl; 
}
