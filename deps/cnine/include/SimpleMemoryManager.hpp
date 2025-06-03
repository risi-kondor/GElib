/*
 * This file is part of cnine, a lightweight C++ tensor library. 
 *  
 * Copyright (c) 2024, Imre Risi Kondor
 *
 * This source code file is subject to the terms of the noncommercial 
 * license distributed with cnine in the file LICENSE.TXT. Commercial 
 * use is prohibited. All redistributed versions of this file (in 
 * original or modified form) must retain this copyright notice and 
 * must be accompanied by a verbatim copy of the license. 
 *
 */


#ifndef _CnineSimpleMemoryManager
#define _CnineSimpleMemoryManager

#include "Cnine_base.hpp"
#include "MemoryManager.hpp"

namespace cnine{


  class SimpleMemoryBlock{
  public:

    size_t beg;
    size_t size;
    bool used=false;

    SimpleMemoryBlock(const size_t& _beg, const size_t& _size):
      beg(_beg), size(_size){}

  };


  class SimpleMemoryManager: public MemoryManager{
  public:
    
    typedef std::list<SimpleMemoryBlock>::iterator block_it;

    size_t _size;
    int dev=0;
    void* arr=nullptr;
    int granularity=128;
    string name;

    mutable std::list<SimpleMemoryBlock> blocks;
    mutable std::unordered_map<void*,block_it> block_map;


    SimpleMemoryManager(const size_t __size, const int _dev=0):
      dev(_dev){
      _size=(__size/granularity)*granularity;
      cout<<"Starting SimpleMemoryManager(size="<<_size<<",dev="<<dev<<")."<<endl;
      if(_size==0) _size=1;
      if(dev==0) arr=::malloc(_size);
      if(dev==1) CUDA_SAFE(cudaMalloc((void **)&arr,_size));
      blocks.push_back(SimpleMemoryBlock(0,_size));
    }

    SimpleMemoryManager(const string _name, const int __size, const int _dev=0):
      SimpleMemoryManager(__size,_dev){
      name=_name;
    }

    ~SimpleMemoryManager(){
      if(dev==0 && arr) {::free(arr);}
      if(dev==1 && arr) {CUDA_SAFE(cudaFree(arr));}
    }


  public: // ---- Copying ------------------------------------------------------------------------------------


    SimpleMemoryManager(const SimpleMemoryManager& x)=delete;


  public: // ---- Access -------------------------------------------------------------------------------------


    size_t size() const{
      return _size;
    }


    template<typename TYPE>
    TYPE* alloc(const int n) const{
      return static_cast<TYPE*>(malloc(n*sizeof(TYPE)));
    }


    void* malloc(const int _n) const{

      int n=roundup(_n,granularity);
      block_it it=blocks.begin();
      while(it!=blocks.end() && (it->used || it->size<n)){
	it++;
      }
      if(it==blocks.end())
	throw std::runtime_error("Memory manager "+name+": out of space.");

      auto& block=*it;
      if(n<block.size)
	blocks.insert(std::next(it),SimpleMemoryBlock(block.beg+n,block.size-n));
      block.size=n;

      block.used=true;
      void* p=static_cast<void*>(static_cast<char*>(arr)+block.beg);
      block_map[p]=it;
      return p;
    }


    void free(void* _p) const{

      void* p=static_cast<void*>(_p);
      auto itt=block_map.find(p);
      //if(itt==block_map.end())
      //throw std::runtime_error("Memory manager "+name+" in free(void*): not a managed object or already deallocated.");
      block_map.erase(p);

      block_it it=itt->second;
      it->used=false;
      
      if(it!=blocks.end()){
	block_it next=std::next(it);
	if(!next->used){
	  it->size+=next->size;
	  blocks.erase(next);
	}
      }

      if(it!=blocks.begin()){
	block_it prev=std::prev(it);
	if(!prev->used){
	  prev->size+=it->size;
	  blocks.erase(it);
	}
      }

    }

    void clear() const{
      blocks.clear();
      block_map.clear();
      blocks.push_back(SimpleMemoryBlock(0,_size));
    }


  private: 

    //block_it prev_block(block_it it) const{
    //if(it==blocks.begin()) return blocks.end();
    //return --it;
    //}
      
    //block_it next_block(block_it it) const{
    //if(it==blocks.end()) return blocks.end();
    //return ++it;
    //}


  public: // ---- I/O ---------------------------------------------------------------------------------------


    string str(const string indent="") const{
      ostringstream oss;
      oss<<indent<<"Memory manager "<<name<<" (size="<<size()<<"):"<<endl;
      //oss<<",addr="<<to_string(static_cast<int>(arr))<<"):"<<endl;
      for(auto& p:blocks){
	oss<<indent<<"  ("<<p.beg<<","<<p.beg+p.size<<"):";
	if(p.used) oss<<" used";
	oss<<endl;
      }
      return oss.str();
    }

    friend ostream& operator<<(ostream& stream, const SimpleMemoryManager& x){
      stream<<x.str(); return stream;
    }


  };


}

#endif 

