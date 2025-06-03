/*
 * This file is part of Cengine, an asynchronous C++/CUDA compute engine. 
 *  
 * Copyright (c) 2020- Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */
#ifndef _Cnine_mtensor_add_mprod_ops
#define _Cnine_mtensor_add_mprod_ops

//#include "CtensorBpack.hpp"
//#include "CtensorBreducer.hpp"
//#include "BatcherA.hpp"
//#include "mtensor_mprod_signature.hpp"


namespace cnine{

  typedef Cengine::Coperator Coperator;
  typedef Cengine::Cnode Cnode;


  template<typename TYPE, int Tsel, int Csel>
  class mtensor_add_mprod_op: public Coperator, public Cengine::CumulativeOperator, public Cengine::InPlaceOperator{
    //			      public Cengine::BatchedOperator, public Cengine::RbatchedOperator{
  public:

    Gdims dims1;
    Gdims dims2;

    mtensor_add_mprod_op(Cnode* R, Cnode* A, Cnode* B, const Gdims& _dims1, const Gdims& _dims2):
      Coperator(R,A,B), dims1(_dims1), dims2(_dims2){}

    static string classname(){
      if(Tsel==0) return "mtensor_add_Mprod<"+to_string(Csel)+">";
      if(Tsel==1) return "mtensor_add_Mprod_TA<"+to_string(Csel)+">";
      if(Tsel==2) return "mtensor_add_Mprod_AT<"+to_string(Csel)+">";
    }

    
  public:

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      auto& obj=MTENSOR(owner); 
      if(Tsel==0) obj.add_mprod(MTENSOR(inputs[1]),MTENSOR(inputs[2]));
      //if(Tsel==1) obj.add_mprod_TA<Csel>(MTENSOR(inputs[1]),MTENSOR(inputs[2]));
      //if(Tsel==2) obj.add_mprod_AT<Csel>(MTENSOR(inputs[1]),MTENSOR(inputs[2]));
    }

  };



  /*
  template<int Tsel, int Csel>
  class ctensor_add_Mprod_op: public Coperator, public CumulativeOperator, public InPlaceOperator, 
			      public BatchedOperator, public RbatchedOperator{
  public:

    Gdims dims1;
    Gdims dims2;

    ctensor_add_Mprod_op(Cnode* R, Cnode* A, Cnode* B, const Gdims& _dims1, const Gdims& _dims2):
      Coperator(R,A,B), dims1(_dims1), dims2(_dims2){}

    static string classname(){
      if(Tsel==0) return "ctensor_add_Mprod<"+to_string(Csel)+">";
      if(Tsel==1) return "ctensor_add_Mprod_TA<"+to_string(Csel)+">";
      if(Tsel==2) return "ctensor_add_Mprod_AT<"+to_string(Csel)+">";
    }
    
  public:

    virtual void exec(){
      assert(!owner->obj);
      owner->obj=inputs[0]->obj;
      CtensorB& obj=MTENSOR(owner); 
      if(Tsel==0) obj.add_Mprod<Csel>(MTENSOR(inputs[1]),MTENSOR(inputs[2]));
      if(Tsel==1) obj.add_Mprod_TA<Csel>(MTENSOR(inputs[1]),MTENSOR(inputs[2]));
      if(Tsel==2) obj.add_Mprod_AT<Csel>(MTENSOR(inputs[1]),MTENSOR(inputs[2]));
    }


    virtual void batched_exec(const vector<Cnode*>& nodes ){
      assert(nodes.size()>0);
      BasicCnodeEngine* engine=nodes[0]->engine;
      const int N=nodes.size();

      CtensorBpack R(nodes,0);
      CtensorBpack X(nodes,1);
      CtensorBpack Y(nodes,2);

      if(Tsel==0) R.add_Mprod<Csel>(X,Y);
      if(Tsel==1) R.add_Mprod_TA<Csel>(X,Y);
      if(Tsel==2) R.add_Mprod_AT<Csel>(X,Y);

      for(int i=0; i<N; i++)
      nodes[i]->op->owner->obj=R.pack[i];

      for(int i=0; i<N; i++){
	engine->done(nodes[i]);
      }
    }


    virtual void rbatched_exec(const vector<Cnode*>& nodes){
      assert(nodes.size()>0);
      const int N=nodes.size();
      int dev=CTENSORB(nodes[0]->op->inputs[0]).device;
      assert(dev==1);

      CtensorBreducer R(N,CTENSORB(nodes[0]));
      CtensorBpack X(nodes,1);
      CtensorBpack Y(nodes,2);
      
      if(Tsel==0) R.add_Mprod<Csel>(X,Y);
      if(Tsel==1) R.add_Mprod_TA<Csel>(X,Y);
      if(Tsel==2) R.add_Mprod_AT<Csel>(X,Y);
    }


  public:

    string str() const{
      return "ctensor_add_Mprod"+inp_str();
    }

    static int _batcher_id;
    int batcher_id() const{return _batcher_id;}
    void set_batcher_id(const int i){_batcher_id=i;}
    string batcher_name() const{
      if(Tsel==0) return "ctensor_add_Mprod<"+to_string(Csel)+">"+signature().str();
      if(Tsel==1) return "ctensor_add_Mprod_TA<"+to_string(Csel)+">"+signature().str();
      if(Tsel==2) return "ctensor_add_Mprod_AT<"+to_string(Csel)+">"+signature().str();
    }
    ctensor_Mprod_signature signature() const{
      return ctensor_Mprod_signature(dims1,dims2);}
    Batcher* spawn_batcher() const{
      return new MetaBatcher<ctensor_add_Mprod_op,ctensor_Mprod_signature,BatcherA<ctensor_add_Mprod_op<Tsel,Csel> > >(inputs[0]->engine);
    }


    static int _rbatcher_id;
    void set_rbatcher_id(const int i){_rbatcher_id=i;}
    int rbatcher_id() const{return _rbatcher_id;}
    string rbatcher_name() const{
      if(Tsel==0) return "ctensor_add_Mprod<"+to_string(Csel)+">"+signature().str();
      if(Tsel==1) return "ctensor_add_Mprod_TA<"+to_string(Csel)+">"+signature().str();
      if(Tsel==2) return "ctensor_add_Mprod_AT<"+to_string(Csel)+">"+signature().str();
    }
    ctensor_Mprod_signature rsignature() const{
      return ctensor_Mprod_signature(dims1,dims2);}
    Rbatcher_base* spawn_rbatcher(BasicCnodeEngine* engine) const{
      return new MetaRbatcher<ctensor_add_Mprod_op,ctensor_Mprod_signature,Rbatcher>(engine);
    }
    
  };
  */

}


#endif
