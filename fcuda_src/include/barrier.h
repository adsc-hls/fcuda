//============================================================================//
//    FCUDA
//    Copyright (c) <2016> 
//    <University of Illinois at Urbana-Champaign>
//    <University of California at Los Angeles> 
//    All rights reserved.
// 
//    Developed by:
// 
//        <ES CAD Group & IMPACT Research Group>
//            <University of Illinois at Urbana-Champaign>
//            <http://dchen.ece.illinois.edu/>
//            <http://impact.crhc.illinois.edu/>
// 
//        <VAST Laboratory>
//            <University of California at Los Angeles>
//            <http://vast.cs.ucla.edu/>
// 
//        <Hardware Research Group>
//            <Advanced Digital Sciences Center>
//            <http://adsc.illinois.edu/>
//============================================================================//

#ifndef MCUDA_PTHREAD_BARRIERS
#define MCUDA_PTHREAD_BARRIERS

/*
 * Barriers
 */
#include <pthread.h>


/* 
 * The pthread barriers aren't fully portable, so we've built our own
 */
typedef struct {
  pthread_mutex_t lock;
  pthread_mutex_t queue;
  int count;
  int limit;
} pthread_barrier_mcuda;

static void 
pthread_barrier_init_mcuda(pthread_barrier_mcuda *b, int limit) {
  b->count = 0;
  b->limit = limit;
  pthread_mutex_init(&b->queue, NULL);
  pthread_mutex_lock(&b->queue);
  pthread_mutex_init(&b->lock, NULL);
}

static void 
pthread_barrier_wait_mcuda(pthread_barrier_mcuda *b) {
  pthread_mutex_lock(&b->lock);
//  printf("Entering barrier %d: count = %d\n", b, b->count);
  if (b->count < b->limit-1) {
    /* delay */
    b->count++;
    pthread_mutex_unlock(&b->lock);
    pthread_mutex_lock(&b->queue);
  }
  /* continue */
  if (b->count == 0) pthread_mutex_unlock(&b->lock);
  else {
    b->count--;
    pthread_mutex_unlock(&b->queue);
  }
//  printf("Leaving barrier %d\n", b);
}

#endif
