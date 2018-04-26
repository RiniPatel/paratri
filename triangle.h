#ifndef TRIANGLE_REF_H

#include <cstdint>



unsigned int count_triangles_orig(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t * triangle_list);

#if OMP
#include <omp.h>
unsigned int count_triangles_omp(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t * triangle_list);
#endif
#define TRIANGLE_REF_H
#endif
