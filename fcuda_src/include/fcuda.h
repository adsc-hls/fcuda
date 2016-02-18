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
