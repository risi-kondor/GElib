
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


#ifndef _ThreadPool
#define _ThreadPool

namespace cnine{

  class ThreadPool{
  public:

    mutex mx;
    mutex gate;
    atomic<int> nthreads;
    int maxthreads=4;
    vector<thread> threads;


  public:

    ThreadPool()=delete;
  
    ThreadPool(const int _maxthreads=4): 
      maxthreads(_maxthreads){}

    ~ThreadPool(){
      while(nrunning>0){
	running_cv.wait();
      }
    }


  public:

    template<class FUNCTION>
    launch(FUNCTION lambda){
      
    }


    template<class FUNCTION, class OBJ>
    inline void ThreadPool::add(FUNCTION lambda, const OBJ x){
      lock_guard<mutex> lock(mx); //                                   unnecessary if called from a single thread
      threadManager.enqueue(this);
      gate.lock(); //                                                  gate can only be unlocked by threadManager
      nthreads++;
      threads.push_back(thread([this,lambda](OBJ _x){
	    lambda(_x); 
	    nthreads--; threadManager.release(this);},x));
      #ifdef _THREADBANKVERBOSE
      printinfo();
      #endif
 }



    bool is_ready(){return nthreads<maxthreads;}

    void printinfo();

  
  };


}


#endif 
