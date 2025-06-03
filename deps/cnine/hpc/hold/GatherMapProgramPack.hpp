/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2021, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */

#ifndef _GatherMapProgramPack
#define _GatherMapProgramPack

#include "Cnine_base.hpp"
#include "object_pack.hpp"
#include "GatherMapProgram.hpp"
#include "MultiLoop.hpp"


namespace cnine{

  extern GPUbuffer<float>  GatherRowsMulti_fbuf;


  class GatherMapProgramPack: public shared_object_pack<GatherMapProgram>{
  public:
    
    

  public: // ---- Execution ----------------------------------------------------------------------------------


    template<typename TYPE>
    void operator()(const Ltensor<TYPE>& output, const Ltensor<TYPE>& arg0) const{

      int N=size();
      CNINE_ASSRT(N>0);
      CNINE_ASSRT(output.get_dev()==arg0.get_dev());
      CNINE_ASSRT(arg0.ndims()==2);
      CNINE_ASSRT(output.ndims()==2);
      int nc=arg0.dim(1);
      int dev=output.get_dev();

      GatherMapProgram& first=*obj[0];
      vector<Ltensor<TYPE>*> v(first.vars.size());
      v[0]=new Ltensor<TYPE>(arg0);
      v[1]=new Ltensor<TYPE>(output);

      int Nvars=first.vars.size();
      Ltensor<int> offsets({Nvars,N+1},0);
      for(int i=0; i<Nvars; i++){
	int t=0;
	for(int j=0; j<N; j++){
	  offsets.set(i,j,t); 
	  t+=obj[j]->vars[i].dims[0];
	}
	offsets.set(i,N,t);
      }

      LtensorView<TYPE>* view_from_buffer=nullptr; 
      if(dev==1 && first.vars.size()==3){
	int ncols=nc*first.vars[2].dims[1];
	if(first.is_inverse) ncols=output.dim(1)*first.vars[2].dims[1];
	auto& buf=GatherRowsMulti_fbuf;
	buf.reset(offsets(2,N)*ncols);
	view_from_buffer=new LtensorView<TYPE>(buf.arr,buf.dev,Gdims(offsets(2,N),ncols));
	view_from_buffer->set_zero();
	//CUDA_SAFE(cudaDeviceSynchronize());
      }

      for(int i=2; i<first.vars.size(); i++){
	int ncols=nc*first.vars[i].dims[1];
	if(first.is_inverse) ncols=output.dim(1)*first.vars[i].dims[1];
	if(i==2 && view_from_buffer) 
	  v[i]=new Ltensor<TYPE>(*view_from_buffer);
	else
	  v[i]=new Ltensor<TYPE>(Gdims(offsets(i,N),ncols),0,dev);
      }

      if(dev==0){
      MultiLoop(N,[&](const int j){
	  for(auto& p:obj[j]->instructions){
	    CNINE_ASSRT(p.out<v.size());
	    CNINE_ASSRT(p.in<v.size());
	    int ooffs=offsets(p.out,j);
	    int orows=offsets(p.out,j+1)-offsets(p.out,j);
	    Ltensor<TYPE> out(v[p.out]->rows(ooffs,orows));
	    int ioffs=offsets(p.in,j);
	    int irows=offsets(p.in,j+1)-offsets(p.in,j);
	    Ltensor<TYPE> in(v[p.in]->rows(ioffs,irows));
	    GatherRows()(out,in,*p.map);
	  }
	});
      }

      if(dev==1){
	int Ninstr=first.instructions.size();
	for(int i=0; i<Ninstr; i++){
	  auto& instr=first.instructions[i];
	  vector<shared_ptr<const GatherMapB> > maps;
	  for(int j=0; j<N; j++)
	    maps.push_back(obj[j]->instructions[i].map);
	  GatherRowsMulti()(*v[instr.out],*v[instr.in],maps,
	    Ltensor<int>(offsets.row(instr.out))*instr.map->out_columns,
	    Ltensor<int>(offsets.row(instr.in))*instr.map->in_columns);
	}
      }

      for(auto p:v)
	delete p;
      if(view_from_buffer) delete view_from_buffer;

    }



  };

}


#endif 
