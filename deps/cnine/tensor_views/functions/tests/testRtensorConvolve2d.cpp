#include "Cnine_base.cpp"
#include "RtensorA.hpp"
#include "CnineSession.hpp"
#include "RtensorConvolve2d.hpp"
#include "RtensorConvolve2dSparse.hpp"

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
  int padding=1;

  bool sparse=false;

  if(true){
    cout<<"3D case"<<endl;

    RtensorA x=RtensorA::zero({nx,nx,1});
    x.set(1,2,0,1.0);
    cout<<x.view3().slice2(0)<<endl;

    RtensorA w=RtensorA::sequential({nout,nw,nw,1});
    cout<<w.view4().slice0(0).slice2(0)<<endl;
    
    auto r=convolve2D(x,w,padding,padding);
    cout<<r.view3().slice2(0)<<endl;

    #ifdef _WITH_CUDA
    //cout<<"GPU"<<endl;
    //RtensorA xg(x,1);
    //RtensorA wg(w,1);
    //auto rg=convolve2D(xg,wg,padding,padding);
    //cout<<rg.to_device(0).view3().slice2(0)<<endl;
    #endif 
    cout<<endl;

    if(sparse){
      CSRmatrix<float> ws(w.view4().fuse23().fuse12());
      cout<<ws<<endl;
      auto rs=convolve2D(x,ws,nw,nw);
      cout<<rs.view3().slice2(0)<<endl;
    }


  }


  if(true){
    cout<<"4D case"<<endl;

    RtensorA x=RtensorA::zero({nx,nx,1,nc});
    x.set(1,2,0,0,1.0);
    cout<<x.view4().slice3(0).slice2(0)<<endl;
    
    RtensorA w=RtensorA::sequential({nout,nw,nw,1});
    cout<<w.view4().slice0(0).slice2(0)<<endl;
    
    auto r=convolve2D(x,w,padding,padding);
    cout<<r.view4().slice3(0).slice2(0)<<endl;

    #ifdef _WITH_CUDA
    cout<<"GPU"<<endl;
    RtensorA xg(x,1);
    RtensorA wg(w,1);
    auto rg=convolve2D(xg,wg,padding,padding);
    cout<<rg.to_device(0).view4().slice3(0).slice2(0)<<endl;
    #endif 
    cout<<endl;

    if(sparse){
      CSRmatrix<float> ws(w.view4().fuse23().fuse12());
      cout<<ws<<endl;
      auto rs=convolve2D(x,ws,nw,nw);
      cout<<rs.view4().slice3(0).slice2(0)<<endl;
     }

    cout<<endl;
  }


  if(true){
    cout<<"5D case"<<endl;

    RtensorA x=RtensorA::zero({nb,nx,nx,1,nc});
    x.set(0,1,2,0,0,1.0);
    cout<<x.view5().slice0(0).slice3(0).slice2(0)<<endl;
    
    RtensorA w=RtensorA::sequential({nout,nw,nw,1});
    cout<<w.view4().slice0(0).slice2(0)<<endl;
    
    auto r=convolve2D(x,w,padding,padding);
    cout<<r.view5().slice0(0).slice3(0).slice2(0)<<endl;

    #ifdef _WITH_CUDA
    cout<<"GPU"<<endl;
    RtensorA xg(x,1);
    RtensorA wg(w,1);
    auto rg=convolve2D(xg,wg,padding,padding);
    cout<<rg.to_device(0).view5().slice0(0).slice3(0).slice2(0)<<endl;
    #endif 

    if(sparse){
      CSRmatrix<float> ws(w.view4().fuse23().fuse12());
      cout<<ws<<endl;
      auto rs=convolve2D(x,ws,nw,nw);
      cout<<rs.view5().slice0(0).slice3(0).slice2(0)<<endl;
    }

    cout<<endl;
  }


  if(false){
    cout<<"6D case"<<endl;

    RtensorA x=RtensorA::zero({nb,nx,nx,nd,1,nc});
    x.set({0,1,2,0,0,0},1.0);
    cout<<x.view6().slice0(0).slice2(0).slice2(0).slice2(0)<<endl;
    
    RtensorA w=RtensorA::sequential({nout,nw,nw,1});
    cout<<w.view4().slice0(0).slice2(0)<<endl;
    CSRmatrix<float> ws(w.view4().fuse23().fuse12());
    cout<<ws<<endl;
    
    auto r=convolve2D(x,w);
    cout<<r.view6().slice0(0).slice2(0).slice2(0).slice2(0)<<endl;

    auto rs=convolve2D(x,ws,nw,nw);
    cout<<rs.view6().slice0(0).slice2(0).slice2(0).slice2(0)<<endl;

    cout<<endl;
  }



  cout<<endl;


  

}
    //RtensorA r=RtensorA::zero({nx-nw+1+2*padding,nx-nw+1+2*padding,nout});
    //RtensorConvolve2d()(r.view5(),x.view5(),w.view4());
