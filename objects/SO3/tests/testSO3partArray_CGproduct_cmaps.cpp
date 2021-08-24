#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"

#include "CellwiseBinaryCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "OuterCmap.hpp"
#include "Convolve2Cmap.hpp"

#include "SO3partA_CGproduct_cop.hpp"

#include "InnerCmap.hpp"
#include "MVprodCmap.hpp"
#include "Convolve2Cmap.hpp"

using namespace cnine;
using namespace GElib;

typedef SO3partA_CGproduct_cop SO3part_CGproduct;


int main(int argc, char** argv){
  GElibSession session;
  device dev=deviceid::GPU0;
  cout<<endl;

  Gdims adims({2,2});

  SO3partArray u(adims,2,2,fill::gaussian);
  SO3partArray v(adims,2,2,fill::gaussian);
  SO3partArray w(dims(2),2,2,fill::gaussian);
  SO3partArray z(dims(3,3),2,2,fill::gaussian);
  SO3part a(2,2,fill::gaussian);
  printl("u",u);
  printl("v",v);

  SO3partArray r0(dims(1,1),2,4,fill::zero);
  add_inner<SO3part_CGproduct>(r0,u,v);
  printl("add_inner<SO3part_CGproduct>(r0,u,v)",r0);

  SO3partArray r1(dims(2),2,4,fill::zero);
  add_MVprod<SO3part_CGproduct>(r1,v,w);
  printl("add_MVprod<SO3part_CGproduct>(r1,v,w)",r1);

  SO3partArray r2(adims,2,4,fill::zero);
  add_convolve2<SO3part_CGproduct>(r2,z,v);
  printl("add_convolve2<SO3part_CGproduct>(r2,z,v)",r2);


#ifdef _WITH_CUDA 

  SO3partArray ug=u.to(dev);
  SO3partArray vg=v.to(dev);
  SO3partArray wg=w.to(dev);
  SO3partArray zg=z.to(dev);
  SO3part ag=a.to(dev);
  cout<<"--------"<<endl;

  SO3partArray r0g(dims(1,1),2,4,fill::zero,dev);
  add_inner<SO3part_CGproduct>(r0g,ug,vg);
  printl("add_inner<SO3part_CGproduct>(r0g,ug,vg)",r0g);

  SO3partArray r1g(dims(2),2,4,fill::zero,dev);
  add_MVprod<SO3part_CGproduct>(r1g,vg,wg);
  printl("add_MVprod<SO3part_CGproduct>(r1g,vg,wg)",r1g);

  SO3partArray r2g(adims,2,4,fill::zero,dev);
  add_convolve2<SO3part_CGproduct>(r2g,zg,vg);
  printl("add_convolve2<SO3part_CGproduct>(r2g,zg,vg)",r2g);

#endif 

  cout<<endl; 
}
