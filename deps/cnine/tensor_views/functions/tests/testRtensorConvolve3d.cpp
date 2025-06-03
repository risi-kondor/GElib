#include "Cnine_base.cpp"
#include "RtensorA.hpp"
#include "CnineSession.hpp"
#include "RtensorConvolve3d.hpp"
#include "RtensorConvolve3dSparse.hpp"
#include "CtensorConvolve3d.hpp"
#include "CtensorConvolve3d_back0.hpp"

using namespace cnine;

typedef RtensorA rtensor; 


int main(int argc, char** argv){

  cnine_session session;

  int nb=1;
  int nx=5;
  int nw=3;
  int nc=1;
  int nd=1;
  int nout=1;
  int padding=0;

  bool sparse=true;

  if(true){
    cout<<"4D case"<<endl;

    RtensorA x=RtensorA::zero({nx,nx,nx,1});
    x.set(1,2,2,0,1.0);
    cout<<x.view4().slice3(0).slice2(2)<<endl;

    RtensorA w=RtensorA::sequential({nout,nw,nw,nw,1});
    cout<<w.view5().slice0(0).slice3(0).slice2(0)<<endl;
    
    auto r=convolve3D(x,w,padding,padding,padding);
    cout<<r.view4().slice3(0).slice2(1)<<endl;

    #ifdef _WITH_CUDA
    cout<<"GPU"<<endl;
    RtensorA xg(x,1);
    RtensorA wg(w,1);
    auto rg=convolve3D(xg,wg,padding,padding,padding);
    cout<<rg.to_device(0).view4().slice3(0).slice2(1)<<endl;
    #endif 
    cout<<endl;

    if(sparse){
      CSRmatrix<float> ws(w.view5().fuse34().fuse23().fuse12());
      //cout<<ws<<endl;
      auto rs=convolve3D(x,ws,nw,nw,nw,padding,padding,padding);
      cout<<rs.view4().slice3(0).slice2(1)<<endl;

      #ifdef _WITH_CUDA
      cout<<"GPU"<<endl;
      RtensorA xg(x,1);
      CSRmatrix wg(ws,1);
      auto rsg=convolve3D(xg,wg,nw,nw,nw,padding,padding,padding);
      cout<<rsg.to_device(0).view4().slice3(0).slice2(1)<<endl;
      #endif 
    cout<<endl;
    }

  }


  if(true){
    cout<<"5D case"<<endl;

    RtensorA x=RtensorA::zero({nx,nx,nx,1,nc});
    x.set(1,2,2,0,0,1.0);
    cout<<x.view5().slice4(0).slice3(0).slice2(2)<<endl;
    
    RtensorA w=RtensorA::sequential({nout,nw,nw,nw,1});
    cout<<w.view5().slice0(0).slice3(0).slice2(0)<<endl;
    
    auto r=convolve3D(x,w,padding,padding,padding);
    cout<<r.view5().slice4(0).slice3(0).slice2(1)<<endl;
    
    #ifdef _WITH_CUDA
    cout<<"GPU"<<endl;
    RtensorA xg(x,1);
    RtensorA wg(w,1);
    auto rg=convolve3D(xg,wg,padding,padding,padding);
    cout<<rg.to_device(0).view5().slice4(0).slice3(0).slice2(1)<<endl;
    #endif 
    cout<<endl;
    
    if(sparse){
      CSRmatrix<float> ws(w.view5().fuse34().fuse23().fuse12());
      //cout<<ws<<endl;
      auto rs=convolve3D(x,ws,nw,nw,nw,padding,padding,padding);
      cout<<rs.view5().slice4(0).slice3(0).slice2(1)<<endl;

     #ifdef _WITH_CUDA
      cout<<"GPU"<<endl;
      RtensorA xg(x,1);
      CSRmatrix wg(ws,1);
      auto rsg=convolve3D(xg,wg,nw,nw,nw,padding,padding,padding);
      cout<<rsg.to_device(0).view5().slice4(0).slice3(0).slice2(1)<<endl;
      #endif 
    }

    cout<<endl;
  }


  if(true){
    cout<<"6D case"<<endl;

    RtensorA x=RtensorA::zero({nb,nx,nx,nx,1,nc});
    x.set({0,1,2,2,0,0},1.0);
    cout<<x.view6().slice0(0).slice4(0).slice3(0).slice2(2)<<endl;
    
    RtensorA w=RtensorA::sequential({nout,nw,nw,nw,1});
    cout<<w.view5().slice0(0).slice3(0).slice2(0)<<endl;
    
    auto r=convolve3D(x,w,padding,padding,padding);
    cout<<r.view6().slice0(0).slice4(0).slice3(0).slice2(1)<<endl;

    #ifdef _WITH_CUDA
    cout<<"GPU"<<endl;
    RtensorA xg(x,1);
    RtensorA wg(w,1);
    auto rg=convolve3D(xg,wg,padding,padding,padding);
    cout<<rg.to_device(0).view6().slice0(0).slice4(0).slice3(0).slice2(1)<<endl;
    #endif 

    if(sparse){
      CSRmatrix<float> ws(w.view5().fuse34().fuse23().fuse12());
      //cout<<ws<<endl;
      auto rs=convolve3D(x,ws,nw,nw,nw,padding,padding,padding);
      cout<<rs.view6().slice0(0).slice4(0).slice3(0).slice2(1)<<endl;

     #ifdef _WITH_CUDA
      cout<<"GPU"<<endl;
      RtensorA xg(x,1);
      CSRmatrix wg(ws,1);
      auto rsg=convolve3D(xg,wg,nw,nw,nw,padding,padding,padding);
      cout<<rsg.to_device(0).view6().slice0(0).slice4(0).slice3(0).slice2(1)<<endl;
      #endif 
    }

    cout<<endl;
  }


  cout<<endl;
  
}
