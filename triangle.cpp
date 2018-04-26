#include "triangle.h"

#include <stdio.h>

/*
This function computes the number of triangles in the graph

Description of algorithm:

1. A triangle is created with 3 vertices (x, y, z)
2. We assume that the vertices are numbered from 0 to N-1
3. The algorithm iterates over the vertices, and for each vertex (y) with a number i,
            the algorithm checks if there is a triangle (x, y, z) where
                x has a smaller number than i, and z has a larger number than i.
*/
unsigned int count_triangles_orig(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t * triangle_list) {
    uint32_t delta = 0;

    // The outer loop traverses over all the vertices. The iteration starts 
    // with vertex 1
    for (uint32_t i = 1; i < N - 1; i++) {
        uint32_t num_nnz_curr_row_x = IA[i + 1] - IA[i];
        uint32_t *x_col_begin = &JA[IA[i]];
        uint32_t *row_bound = &JA[IA[i + 1]];
        uint32_t *x_col_end = row_bound;

        // printf("%d\n", num_nnz_curr_row_x);
        for (uint32_t idx = 0; idx < num_nnz_curr_row_x; idx++)
        {
            if (x_col_begin[idx] > (i - 1)) {
                x_col_end = &x_col_begin[idx];
                break;
            }
        }

        uint32_t num_nnz_y = (row_bound - x_col_end);
        uint32_t num_nnz_x = (x_col_end - x_col_begin);

        // This is where the real triangle counting begins.
        // We search through all possible vertices for x
        for (uint32_t j = 0; j < num_nnz_y; ++j) {
            uint32_t *x_col = x_col_begin;
            uint32_t *A_col = JA + IA[x_col_end[j]];
            uint32_t *A_col_max = JA + IA[x_col_end[j] + 1];

            // this loop searches through all possible vertices for z.
            for (uint32_t k = 0; k < num_nnz_x; ++k) {
                while ((*A_col < x_col[k])  && (A_col < A_col_max)) ++A_col;

                // for triangle enumeration i, *x_col_end, *A_col
                if (*A_col == x_col[k]) {
                    delta += 1;
                    int idx = delta * 3;
                    triangle_list[idx] = i;
                    triangle_list[idx+1] = *A_col;
                    triangle_list[idx+2] = x_col_end[j];
                }
            }
        }
    }

    return delta;
}

#if OMP
uint32_t count_triangles_omp(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t * triangle_list)
{
    uint32_t delta = 0;

    int num_threads = omp_get_max_threads();
    uint32_t prefix_sum_arr[num_threads];

    // The outer loop traverses over all the vertices. The iteration starts 
    // with vertex 1
    #pragma omp parallel
    {
        prefix_sum_arr[omp_get_thread_num()] = 0;

        #pragma omp for schedule(static, 4) reduction(+: delta)
        for (uint32_t i = 1; i < N - 1; i++) {
            uint32_t num_nnz_curr_row_x = IA[i + 1] - IA[i];
            uint32_t *x_col_begin = &JA[IA[i]];
            uint32_t *row_bound = &JA[IA[i + 1]];
            uint32_t *x_col_end = row_bound;

            for (uint32_t idx = 0; idx < num_nnz_curr_row_x; idx++)
            {
                if (x_col_begin[idx] > (i - 1)) {
                    x_col_end = &x_col_begin[idx];
                    break;
                }
            }

            uint32_t num_nnz_y = (row_bound - x_col_end);
            uint32_t num_nnz_x = (x_col_end - x_col_begin);

            // This is where the real triangle counting begins.
            // We search through all possible vertices for x
            for (uint32_t j = 0; j < num_nnz_y; ++j) {
                uint32_t *x_col = x_col_begin;
                uint32_t *A_col = JA + IA[x_col_end[j]];
                uint32_t *A_col_max = JA + IA[x_col_end[j] + 1];

                // this loop searches through all possible vertices for z.
                for (uint32_t k = 0; k < num_nnz_x; ++k) {
                    while ((*A_col < x_col[k])  && (A_col < A_col_max)) ++A_col;

                    // for triangle enumeration i, *x_col_end, *A_col
                    if (*A_col == x_col[k]) {
                        int idx = delta * 3;
                        triangle_list[idx] = i;
                        triangle_list[idx+1] = *A_col;
                        triangle_list[idx+2] = x_col_end[j];
                        delta += 1;
                        prefix_sum_arr[omp_get_thread_num()] += 1;
                    }
                }
            }
        }
    }
    

    int temp = prefix_sum_arr[num_threads-1];
    // upsweep phase
    for (int twod = 1; twod < num_threads; twod*=2)
    {
        int twod1 = twod*2;
        // parallel
        #pragma omp for 
        for (int i = 0; i < num_threads; i += twod1)
        {
            prefix_sum_arr[i+twod1-1] += prefix_sum_arr[i+twod-1];
        }
    }
    prefix_sum_arr[num_threads-1] = 0;

    // downsweep phase
    for (int twod = num_threads/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod*2;
        // parallel
        #pragma omp for 
        for (int i = 0; i < num_threads; i += twod1)
        {
            int tmp = prefix_sum_arr[i+twod-1];
            prefix_sum_arr[i+twod-1] = prefix_sum_arr[i+twod1-1];
            prefix_sum_arr[i+twod1-1] += tmp;
        }
    }

    return prefix_sum_arr[num_threads-1] + temp;
}
#endif
