#include "GElib_base.cpp"
#include "SO3partArray.hpp"
#include "GElibSession.hpp"

#include "CellwiseBinaryCmap.hpp"
#include "BroadcastBinaryCmap.hpp"
#include "OuterCmap.hpp"
#include "Convolve2Cmap.hpp"

#include "SO3partA_CGproduct_back0_cop.hpp"

using namespace cnine;
using namespace GElib;

typedef SO3partA_CGproduct_back0_cop SO3part_CGproduct_back0;


int main(int argc, char** argv){
  GElibSession session;
  device dev=deviceid::GPU0;
  cout<<endl;

  Gdims adims({2,2});
  
  SO3partArray u(adims,2,2,fill::gaussian);
  SO3partArray v(adims,2,2,fill::gaussian);
  SO3partArray vv(dims(2),2,2,fill::gaussian);
  SO3partArray g(adims,2,4,fill::gaussian);
  SO3partArray gv(dims(2),2,4,fill::gaussian);
  SO3part gm(2,4,fill::gaussian);
  //printl("u",u);
  //printl("v",v);

  SO3partArray r0(adims,2,2,fill::zero);
  add_cellwise<SO3part_CGproduct_back0>(r0,g,v);
  printl("add_cellwise<SO3part_CGproduct_back0>(r0,g,v)",r0);

  SO3partArray r1(adims,2,2,fill::zero);
  add_broadcast<SO3part_CGproduct_back0>(r1,gm,v);
  printl("add_broadcast<SO3part_CGproduct_back0>(r1,gm,v)",r1);

  SO3partArray r2(adims,2,2,fill::zero);
  add_outer<SO3part_CGproduct_back0>(r2,gv,vv);
  printl("add_outer<SO3part_CGproduct_back0>(r2,gv,vv)",r2);

#ifdef _WITH_CUDA

  SO3partArray ug=u.to(dev);
  SO3partArray vg=v.to(dev);
  SO3partArray vvg=vv.to(dev);
  SO3partArray gg=g.to(dev);
  SO3partArray gvg=gv.to(dev);
  SO3part gmg=gm.to(dev);

  cout<<endl<<"-------------"<<endl<<endl;

  SO3partArray r0g(adims,2,2,fill::zero,dev);
  add_cellwise<SO3part_CGproduct_back0>(r0g,gg,vg);
  printl("add_cellwise<SO3part_CGproduct_back0>(r0,g,v)",r0);
  printl("add_cellwise<SO3part_CGproduct_back0>(r0g,gg,vg)",r0g);

  SO3partArray r1g(adims,2,2,fill::zero,dev);
  add_broadcast<SO3part_CGproduct_back0>(r1g,gmg,vg);
  printl("add_broadcast<SO3part_CGproduct_back0>(r1,gm,v)",r1);
  printl("add_broadcast<SO3part_CGproduct_back0>(r1g,gmg,vg)",r1g);

  SO3partArray r2g(adims,2,2,fill::zero,dev);
  add_outer<SO3part_CGproduct_back0>(r2g,gvg,vvg);
  printl("add_outer<SO3part_CGproduct_back0>(r2,gv,vv)",r2);
  printl("add_outer<SO3part_CGproduct_back0>(r2g,gvg,vvg)",r2g);

#endif 

  cout<<endl; 
}
