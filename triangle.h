#ifndef _TRIANGLE_REF_H
#define _TRIANGLE_REF_H

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

#define MAX_TRIANGLES 20000000

unsigned int count_triangles_orig(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t * triangle_list);

#if CUDA
void printCudaInfo();
uint32_t count_triangles_cuda(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t * triangle_list);
#endif

#if OMP
#include <omp.h>
unsigned int count_triangles_omp(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t * triangle_list);
#endif

#endif  //  _TRIANGLE_REF_H
