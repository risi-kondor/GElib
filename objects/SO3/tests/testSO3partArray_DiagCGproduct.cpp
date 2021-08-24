#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"

#include "CellwiseBinaryCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "OuterCmap.hpp"
#include "Convolve2Cmap.hpp"

#include "SO3partA_DiagCGproduct_cop.hpp"

using namespace cnine;
using namespace GElib;

typedef SO3partA_DiagCGproduct_cop SO3part_DiagCGproduct;


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

  SO3partArray r0(adims,2,2,fill::zero);
  add_cellwise<SO3part_DiagCGproduct>(r0,u,v);
  printl("add_cellwise<SO3part_DiagCGproduct>(r0,u,v)",r0);

  SO3partArray r1(adims,2,2,fill::zero);
  add_broadcast<SO3part_DiagCGproduct>(r1,a,v);
  printl("add_broadcast<SO3part_DiagCGproduct>(r1,a,v)",r1);

  SO3partArray r2(adims,2,2,fill::zero);
  add_outer<SO3part_DiagCGproduct>(r2,w,w);
  printl("add_outer<SO3part_DiagCGproduct>(r2,w,w)",r2);

  //printl("convolve2<DiagCGproduct>(z,u,2)",convolve2<DiagCGproduct>(z,u,2));
  //printl("DiagCGproduct(a,u,2)",DiagCGproduct(a,u,2));

#ifdef _WITH_CUDA 

  SO3partArray ug=u.to(dev);
  SO3partArray vg=v.to(dev);
  SO3partArray wg=w.to(dev);
  SO3partArray zg=z.to(dev);
  SO3part ag=a.to(dev);

  SO3partArray r0g(adims,2,2,fill::zero,dev);
  add_cellwise<SO3part_DiagCGproduct>(r0g,ug,vg);
  printl("add_cellwise<SO3part_DiagCGproduct>(r0g,ug,vg)",r0g);

  SO3partArray r1g(adims,2,2,fill::zero,dev);
  add_broadcast<SO3part_DiagCGproduct>(r1g,ag,vg);
  printl("add_broadcast<SO3part_DiagCGproduct>(r1g,ag,vg)",r1g);

  SO3partArray r2g(adims,2,2,fill::zero,dev);
  add_outer<SO3part_DiagCGproduct>(r2g,wg,wg);
  printl("add_outer<SO3part_DiagCGproduct>(r2g,wg,wg)",r2g);

#endif 

  cout<<endl; 
}
