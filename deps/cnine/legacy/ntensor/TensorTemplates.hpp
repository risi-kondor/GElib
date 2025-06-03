/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2023, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineTensorTemplates
#define _CnineTensorTemplates

namespace cnine{


  inline bool batches_conditionally_eq(const int b0, const int b1, const int b2){
    int t=0;
    if(b0!=1) t=b0;
    if(b1!=1){
      if(t>0){if(b1!=t) return false;}
      else t=b1;
    }
    if(b2!=1){
      if(t>0){if(b2!=t) return false;}
      else t=b2;
    }
    return true;
  }


  template<typename TYPE>
  void reconcile_batches(const TYPE& r, const TYPE& x, const TYPE& y,
    const std::function<void(const TYPE&, const TYPE&, const TYPE&)>& fn,
    const std::function<void(const TYPE&, const TYPE&, const TYPE&)>& rfn){
    int rb=r.getb();
    int xb=x.getb();
    int yb=y.getb();
    if (!batches_conditionally_eq(rb,xb,yb)) CNINE_ERROR("Batch mismatch.");
    int B=max(rb,max(xb,yb));

    if(rb==1){
      if(xb==1 && yb==1) fn(r,x,y);
      else rfn(r.unsqueeze(0).unsqueeze(1),x.unsqueeze(0).unsqueeze(1).cinflate(2,B),
	y.unsqueeze(0).cinflate(1,B).unsqueeze(2));
    }else{
      fn(r,x.cinflate(0,B),y.cinflate(0,B));
    }
  }

  template<typename TYPE>
  void reconcile_barray(const TYPE& r, const TYPE& x, const TYPE& y,
    const std::function<void(const TYPE&, const TYPE&, const TYPE&)>& fn,
    const std::function<void(const TYPE&, const TYPE&, const TYPE&)>& rfn){
    
    CNINE_ASSRT(r.getb()==x.getb());
    CNINE_ASSRT(r.getb()==y.getb());

    int rd=r.nadims();
    int xd=x.nadims();
    int yd=y.nadims();

    if(rd==xd&&rd==yd){
      fn(r.baflatten(),x.baflatten(),y.baflatten());
      return;
    }

    if(rd==1){
      if(xd==2&&yd==1){ // matrix-vector
	CNINE_ASSRT(r.adim(0)==x.adim(0));
	CNINE_ASSRT(y.adim(0)==x.adim(1));
	rfn(r.unsqueeze(2),x,y.unsqueeze(2));
	return;
      }
      if(xd==1&&yd==2){ // vector-matrix 
	CNINE_ASSRT(r.adim(0)==y.adim(1));
	CNINE_ASSRT(x.adim(0)==y.adim(0));
	rfn(r.unsqueeze(1),x.unsqueeze(1),y);
	return;
      }
    }

    CNINE_ERROR("Unable to reconcile batch dimensions.");

    /*
    if(rd==2){
      if(xd==2&&yd==1){ // matrix-vector
	CNINE_ASSRT(r.adim(0)==x.adim(0));
	CNINE_ASSRT(r.adim(1)==x.adim(1));
	CNINE_ASSRT(y.adim(0)==x.adim(1));
	fn(r.baflatten(),x.baflatten(),y.insert_dim(2,x.adim(0)));
      }
      if(xd==1&&yd==2){ // vector-matrix 
	CNINE_ASSRT(r.adim(0)==y.adim(0));
	CNINE_ASSRT(r.adim(1)==y.adim(1));
	CNINE_ASSRT(x.adim(0)==y.adim(0));
	OP(r,x.insert_dim(0,y.adim(1),y));
      }
    }
    */

  }


  template<typename TYPE>
  void reconcile_batched_array(const TYPE& r, const TYPE& x, const TYPE& y,
    const std::function<void(const TYPE&, const TYPE&, const TYPE&)>& fn,
    const std::function<void(const TYPE&, const TYPE&, const TYPE&)>& rfn){
    int rb=r.getb();
    int xb=x.getb();
    int yb=y.getb();
    if (!batches_conditionally_eq(rb,xb,yb)) CNINE_ERROR("Batch mismatch.");
    int B=max(rb,max(xb,yb));

    if(rb==1){
      if(xb==1 && yb==1) {reconcile_barray<TYPE>(r,x,y,fn,rfn); return;}
      else{
	if(r.nadims()==x.nadims()&&r.nadims()==y.nadims()){
	  rfn(r.swap_batch_array().unsqueeze(0).unsqueeze(1),
	    x.swap_batch_array().unsqueeze(0).unsqueeze(1).cinflate(2,B),
	    y.swap_batch_array().unsqueeze(0).cinflate(1,B).unsqueeze(2));
	  return;
	}
      }
    }else{
      reconcile_barray<TYPE>(r,x.cinflate(0,B),y.cinflate(0,B),fn,rfn);
      return;
    }
    CNINE_ERROR("Dimension mismatch");
  }

}


#endif 
