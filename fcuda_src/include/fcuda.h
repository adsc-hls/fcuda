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

#ifndef FCUDA_HEADER
#define FCUDA_HEADER

#include <stdlib.h>
#include "cuda_include/builtin_types.h"
#include "fcutil.h"

#ifdef __FCUDA__
extern dim3 blockIdx;
extern dim3 threadIdx;
extern dim3 gridDim;
extern dim3 blockDim;

void __syncthreads() {;}
void memcpy(void* dst, void* src, int size);
#endif

#endif
