#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"

#include "CellwiseBinaryCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "OuterCmap.hpp"
#include "Convolve2Cmap.hpp"

#include "SO3partA_CGproduct_cop.hpp"

using namespace cnine;
using namespace GElib;

typedef SO3partA_CGproduct_cop SO3part_CGproduct;


int main(int argc, char** argv){
  GElibSession session;
  device dev=deviceid::GPU0;
  cout<<endl;

  int l=5;
  int n=32;

  Gdims adims({2,2});

  SO3partArray u(adims,l,n,fill::gaussian);
  SO3partArray v(adims,l,n,fill::gaussian);
  SO3partArray w(dims(2),l,n,fill::gaussian);
  SO3partArray z(dims(3,3),l,n,fill::gaussian);
  SO3part a(l,n,fill::gaussian);
  //printl("u",u);
  //printl("v",v);

  SO3partArray r0(adims,l,n*n,fill::zero);
  add_cellwise<SO3part_CGproduct>(r0,u,v);
  //printl("add_cellwise<SO3part_CGproduct>(r0,u,v)",r0);

  //SO3partArray r1(adims,l,n*n,fill::zero);
  //add_broadcast<SO3part_CGproduct>(r1,a,v);
  //printl("add_broadcast<SO3part_CGproduct>(r1,a,v)",r1);

  //SO3partArray r2(adims,l,n*n,fill::zero);
  //add_outer<SO3part_CGproduct>(r2,w,w);
  //printl("add_outer<SO3part_CGproduct>(r2,w,w)",r2);

  //printl("convolve2<CGproduct>(z,u,2)",convolve2<CGproduct>(z,u,2));
  //printl("CGproduct(a,u,2)",CGproduct(a,u,2));

#ifdef _WITH_CUDA 

  SO3partArray ug=u.to(dev);
  SO3partArray vg=v.to(dev);
  SO3partArray wg=w.to(dev);
  SO3partArray zg=z.to(dev);
  SO3part ag=a.to(dev);

  SO3partArray r0g(adims,l,n*n,fill::zero,dev);
  add_cellwise<SO3part_CGproduct>(r0g,ug,vg);
  //printl("add_cellwise<SO3part_CGproduct>(r0g,ug,vg)",r0g);
  printl("diff",r0g.to(0)-r0);
  
  //SO3partArray r1g(adims,l,n*n,fill::zero,dev);
  //add_broadcast<SO3part_CGproduct>(r1g,ag,vg);
  //printl("add_broadcast<SO3part_CGproduct>(r1g,ag,vg)",r1g);

  //SO3partArray r2g(adims,l,n*n,fill::zero,dev);
  //add_outer<SO3part_CGproduct>(r2g,wg,wg);
  //printl("add_outer<SO3part_CGproduct>(r2g,wg,wg)",r2g);

#endif 

  cout<<endl; 
}
