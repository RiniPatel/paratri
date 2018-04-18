#include "triangle_ref.h"

unsigned int count_triangles_orig(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t * triangle_list) {
    unsigned int delta = 0;

    //The outer loop traverses over all the vertices. The iteration starts with vertex 1
    for (unsigned int i = 1; i < N - 1; i++) {

        unsigned int *curr_row_x = IA + i;
        unsigned int *curr_row_A = IA + i + 1;
        unsigned int num_nnz_curr_row_x = *curr_row_A - *curr_row_x;
        unsigned int *x_col_begin = (JA + *curr_row_x);
        unsigned int *x_col_end = x_col_begin;
        unsigned int *row_bound = x_col_begin + num_nnz_curr_row_x;
        unsigned int col_x_max = i - 1;

        while (x_col_end < row_bound && *x_col_end < col_x_max) ++x_col_end;


        x_col_end -= (*x_col_end > col_x_max || x_col_end == row_bound);

        unsigned int *y_col_begin = x_col_end + 1;
        unsigned int *y_col_end = row_bound - 1;
        unsigned int num_nnz_y = (y_col_end - y_col_begin) + 1;
        unsigned int num_nnz_x = (x_col_end - x_col_begin) + 1;

        unsigned int y_col_first = i + 1;
        unsigned int x_col_first = 0;
        unsigned int *y_col = y_col_begin;

        for (unsigned int j = 0; j < num_nnz_y; ++j, ++y_col) {
            unsigned int row_index_A = *y_col - y_col_first;
            unsigned int *x_col = x_col_begin;
            unsigned int num_nnz_A = *(curr_row_A + row_index_A + 1) - *(curr_row_A + row_index_A);
            unsigned int *A_col = (JA + * (curr_row_A + row_index_A));
            unsigned int *A_col_max = A_col + num_nnz_A;

            // this loop searches through all possible vertices for z.
            for (unsigned int k = 0; k < num_nnz_x && *A_col <= col_x_max; ++k) {

                unsigned int row_index_x = *x_col - x_col_first;
                while ((*A_col < *x_col) && (A_col < A_col_max)) ++A_col;
                ++x_col;

                if (*A_col == row_index_x) {
                    int idx = delta * 3;
                    triangle_list[idx] = i;             
                    triangle_list[idx + 1] = *A_col;
                    triangle_list[idx + 2] = *y_col;
                    delta += 1;
                }
            }
        }
    }

    return delta;
}
