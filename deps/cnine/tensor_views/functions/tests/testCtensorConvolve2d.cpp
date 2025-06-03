#include "Cnine_base.cpp"
#include "RtensorA.hpp"
#include "CnineSession.hpp"
#include "CtensorConvolve2d.hpp"
//#include "CtensorConvolve2dSparse.hpp"

using namespace cnine;



int main(int argc, char** argv){

  cnine_session session;

  int nb=1;
  int nx=5;
  int nw=3;
  int nc=1;
  int nout=1;
  int padding=0;

  if(true){
    cout<<"3d case"<<endl;

    CtensorB x=CtensorB::zero({nx,nx,1});
    x.set(1,2,0,1.0);
    cout<<x.view3().slice2(0)<<endl;
    
    RtensorA w=RtensorA::sequential({nout,nw,nw,1});
    cout<<w.view4().slice0(0).slice2(0)<<endl;
    CSRmatrix<float> ws(w.view4().fuse23().fuse12());
    //cout<<ws<<endl;
    
    auto r=convolve2D(x,w);
    cout<<r.view3().slice2(0)<<endl;

    //auto rs=convolve2D(x,ws,nw,nw);
    //cout<<rs.view3().slice2(0)<<endl;

    cout<<endl;
  }


  if(true){
    cout<<"4d case"<<endl;

    CtensorB x=CtensorB::zero({nx,nx,1,nc});
    x.set(1,2,0,0,1.0);
    cout<<x.view4().slice3(0).slice2(0)<<endl;
    
    RtensorA w=RtensorA::sequential({nout,nw,nw,1});
    cout<<w.view4().slice0(0).slice2(0)<<endl;
    CSRmatrix<float> ws(w.view4().fuse23().fuse12());
    cout<<ws<<endl;
    
    auto r=convolve2D(x,w);
    cout<<r.view4().slice3(0).slice2(0)<<endl;

    //auto rs=convolve2D(x,ws,nw,nw);
    //cout<<rs.view4().slice3(0).slice2(0)<<endl;

    cout<<endl;
  }


  if(true){
    cout<<"5d case"<<endl;

    CtensorB x=CtensorB::zero({nb,nx,nx,1,nc});
    x.set(0,1,2,0,0,1.0);
    cout<<x.view5().slice0(0).slice3(0).slice2(0)<<endl;
    
    RtensorA w=RtensorA::sequential({nout,nw,nw,1});
    cout<<w.view4().slice0(0).slice2(0)<<endl;
    CSRmatrix<float> ws(w.view4().fuse23().fuse12());
    //cout<<ws<<endl;
    
    auto r=convolve2D(x,w);
    cout<<r.view5().slice0(0).slice3(0).slice2(0)<<endl;

    //auto rs=convolve2D(x,ws,nw,nw);
    //cout<<rs.view5().slice0(0).slice3(0).slice2(0)<<endl;

    cout<<endl;
  }



  cout<<endl;

}


  

