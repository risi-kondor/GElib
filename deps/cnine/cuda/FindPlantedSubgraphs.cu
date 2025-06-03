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

#ifndef _FindPlantedSubgraphs_cu
#define _FindPlantedSubgraphs_cu

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cnine_base.hpp"
#include "CUDAhelpers.hpp"
#include "sparse_graph.hpp"
#include "minivec.hpp"
#include "gpu_array.hpp"


__device__ int ipool_size_of(const int* arr, const int i){
  return arr[i+2]-arr[i+1];
}

__device__ int ipool_get(const int* arr, const int i, const int j){
  return arr[arr[i+1]+j];
}


// unused
/*
__device__ int index_of_smallest(const int* arr, int* buf, int n){
  const int tix=threadIdx.x;
  if(tix<n) buf[tix]=tix;

  while(n>1){
    int m=n/2;
    if(tix<m){
      if(arr[buf[2*tix]]<arr[buf[2*tix+1]])
	buf[tix]=buf[2*tix]; 
      else
	buf[tix]=buf[2*tix+1];
    }
    if(2*m==n) 
      n=m;
    else{
      if(tix==0) buf[m]=buf[2*m];
      n=m+1;
    }
  }
  return buf[0];
}
*/




__global__ FindPlantedSubgraphs_kernel(int* _matches, const int N, const int n, const int Hsize, 
  const int* G, const int* _H, const int* _Htraversal, const int* _parent_of, 
  int* target_arr, int* nfound_arr, int bufsize){
  extern __shared__ unsigned char _shared[]; 

  const int bix=blockIdx.x;
  const int tix=threadIdx.x;
  
  int* Htraversal=reinterpret_cast<int*>(_shared);
  int* parent_of=Htraversal+n;
  int* matching=parent_of+n;
  int* sorted=matching+n;
  int* pseudo_iterators=sorted+n;
  int* H=pseudo_iterators+n;

  load(Htraversal,_Htraversal,n);
  load(parent_of,_parent_of,n);
  load(H,_H,Hsize);

  if(tix<n) matching[tix]=-1;
  __syncthreads();


  if(tix==0){ //single threaded beyond this point

    int level=0;
    int w=Htraversal[0];
    int v=target_arr[bix];
    
    int* matches=_matches+bix*bufsize*n;
    int nmatches=0;
    
    while(level>=0){

      int m1=ipool_size_of(H,w);
      int m2=ipool.size_of(G,v);
      bool success=true;

      // check that every neighbor of w that is already part 
      // of the matching corresponds to a neighbor of v 
      for(int j=0; j<m1; j++){
	int y=ipool_get(H,w,j);
	if(matching[y]==-1) continue;
	int vdash=matching[y];

	bool found=false;
	for(int p=0; p<m2; p++)
	  if(ipool_get(G,v,p)==vdash){
	    found=true;
	    break;
	  }
	if(!found){
	  success=false;
	  break;
	}
      }

      // check that every neighbor of v that is already part 
      // of the matching corresponds to a neighbor of w 
      if(success){
	for(int j=0; j<m2; j++){
	  int vdash=ipool_get(G,v,j);
	  int wdash=-1;
	  for(int p=0; p<n; p++)
	    if(matching[p]==vdash){
	      wdash=p;
	      break;
	    }
	  if(wdash>=0){
	    bool found=false;
	    for(int p=0; p<m1; p++)
	      if(ipool_get(H,w,p)==wdash){
		found=true;
		break;
	      }
	    if(!found){
	      success=false;
	      break;
	    }
	  }
	}
      }

      // if w has been successfully matched to v
      if(success){
	matching[w]=v;
      }

      if(success && level==n-1){
	
	for(int i=0; i<n; i++){
	  int best=N;
	  int ix=0;
	  for(int j=0; j<n; j++)
	    if(matching[j]<best){
	      ix=j;
	      best=matching[j];
	    }
	  matching[ix]=N;
	  sorted[i]=best;
	}

	bool found=false;
	for(int i=0; i<nmatches && !found; i++){
	  found=true;
	  for(int j=0; j<n; j++)
	    if(matches[i*n+j]!=sorted[j]){
	      found=false;
	      break;
	    }
	}

	if(!found){
	  if(nmatches==bufsize){
	    nfound_arr[bix]=-1;
	    return;
	  }
	  nmatches++;
	  for(int i=0; i<n; i++)
	    matches[nmatches*n+i]=sorted[i];
	}

	success=false;
      }

      // if matched and not at the final level, try to descend
      // even farther
      if(success){
	int parentv=matching[parent_of[Htraversal[level+1]]];
	pseudo_iterators[level]=0;
	int m3=ipool_size_of(G,parentv);
	int newv=-1;
	for(int j=0; j<m3; j++){
	  int candidate=ipool_get(G,parentv,j);
	  bool found=false; 
	  for(int p=0; p<n; p++)
	    if(matching[p]==candidate){
	      found=true;
	      break;
	    }
	  if(!found){
	    newv=candidate;
	    pseudo_iterators[level]=j+1;
	    break;
	  }
	}
	if(newv>=0){
	  w=Htraversal[level+1];
	  v=newv;
	  level++;
	}else{
	  success=false;
	}
      }

      // if no match or could not descend farther forced to climb back
      // and find alternative paths
      if(!success){
	matching[w]=-1;
	level--;

	while(level>=0){
	  int neww=Htraversal[level+1];
	  int parentv=matching[parent_of[neww]];
	  CNINE_ASSRT(parentv!=-1);
	  int m3=ipoll_size_of(G,parentv);
	  int newv=-1;
	  for(int j=pseudo_iterators[level]; j<m3; j++){
	    int candidate=ipool_get(G,parentv,j);
	    bool found=false;
	    for(int p=0; p<n; p++)
	      if(matching[p]==candidate){
		found=true;
		break;
	      }
	    if(!found){
	      newv=candidate;
	      pseudo_iterators[level]=j+1;
	      break;
	    }
	  }
	  if(newv!=-1){
	    w=neww;
	    v=newv;
	    level++;
	    break;
	  }
	  matching[Htraversal[level]]=-1;
	  level--;
	}
      }

    }
  }


}
  

namespace cnine{

  TensorView<int> FindPlantedSubgraphs_cu(const int_pool& _G, int_pool& _H, const cudaStream_t& stream){

    int n=_H.getn();
    int N=_G.getn();
    int nmatches=0;
    TensorView<int> matches(Gdims(10,n));

    sparse_graph<int,float,LABEL> Hsg(_H);
    int_tree Htree=Hsg.greedy_spanning_tree().as_int_tree();
    //vector<int> Htraversal=Htree.depth_first_traversal();
    gpu_array<int> Htraversal(Htree.depth_first_traversal());
    vector<int> _parent_of(n);
    Htree.traverse([&](const int_tree::node& x){
	if(x.parent()>=0) 
	  parent_of[x.label()]=Htree.node_at(x.parent()).label();
      });
    gpu_array<int> parent_of(_parent_of);
    gpu_array<int> G(_G);
    gpu_array<int> H(_H);


    minivec<int> target(N);
    for(int i=0; i<N; i++) 
      target[i]=i;
    int nremaining=n;
    int bufsize=64;

    while(nremaining>0){

      Tensor<int> buf(Gdims(nremaining,bufsize,n),cnine::fill_raw(),1);
      minivec<int> nfound(nremaining,0,1);
      target.move_to_device(1);

      FindPlantedSubgraphs_kernel<<<nremaining,32,0,stream>>>(buf,N,n,H.memsize,G,H,
	Htraversal,parent_of,target.arr,nfound.arr,bufsize);
      nfound.move_to_device(0);
      target.move_to_device(0);

      int nfailed=0;
      for(int i=0; i<nremaining; i++){
	if(nfound[i]>=0){
	  if(nmatches+nfound[i]>matches.dim(0))
	    matches.resize0(std::max(5,2*(nmatches+nfound[i])));
	  nmatches.rows(nmatches,nfound[i])=buf.slice0(i).rows(0,nfound[i]);
	  nmatches+=nfound[i];
	}else{
	  target[nfailed++]=target[i];
	}
      }
      nremaining=nfailed;
      bufsize*=2;
    }

    return SortRowsUnique(matches.move_to_device(0));
  }


}

#endif 

//__global__ FindPlantedSubgraphs_kernel(int* _matches, const int N, const int n, const int Hsize, 
//  const int* Garr, const int* Harr, 
//  const int* Htraversal, const int* parent_of, int* target_arr, int* nfound_arr, int bufsize){
