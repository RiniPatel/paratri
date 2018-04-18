/**
 * 15-618 Project
 * @author Tushar Goyal (tgoyal1)
 * @author Rini Patel (rinip)
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include "triangle_ref.h"
#include "cycletimer.h"

using namespace std;

#define NUM_ITER 10

/**
 * This function reads the input data. 
 * It populates the IA and JA arrays used to store the adjacency matrix of 
 * the graph in CSR (compressed sparse row) format. 
 * The CSR is a sparse matrix format can be described with 3 arrays (A, IA, JA).
 * 0. The length of IA is the number of vertices in the graph + 1.
 * 1. IA is an array that splits the JA array into multiple subarrays. 
 * Specifically, the (IA[k+1] - IA[k]) is the length of the k^{th} subarray of JA.
 * 2. The k^{th} subarray of JA is the neighbor list for the k^{th} vertex.
 */
int fill_input(string file_name, uint32_t *IA, uint32_t *JA)
{
	FILE * file_IA = fopen((file_name + "_IA.txt").c_str(), "r");
	FILE * file_JA = fopen((file_name + "_JA.txt").c_str(), "r");

	if (file_IA == NULL || file_JA == NULL) {
		return -1;
	}

	int i = 0;
	while (!feof(file_IA)) {
		fscanf(file_IA, "%d\n", &IA[i++]);
	}

	i = 0;
	while (!feof(file_JA)) {
		fscanf(file_JA, "%d\n", &JA[i++]);
	}

	fclose(file_IA);
	fclose(file_JA);

	return 0;
}

/*
This function computes the number of triangles in the graph

Description of algorithm:

1. A triangle is created with 3 vertices (x, y, z)
2. We assume that the vertices are numbered from 0 to N-1
3. The algorithm iterates over the vertices, and for each vertex (y) with a number i,
			the algorithm checks if there is a triangle (x, y, z) where
				x has a smaller number than i, and z has a larger number than i.
*/
uint32_t count_triangles(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A)
{
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
			for (uint32_t k = 0; k < num_nnz_x && *A_col <= (i - 1); ++k) {
				while ((*A_col < x_col[k])	&& (A_col < A_col_max)) ++A_col;
				delta += (*A_col == x_col[k]);

				// for triangle enumeration i, *x_col_end, *A_col
			}
		}
	}

	return delta;
}

void help()
{
	printf("Invalid arguments\n");
	printf("Usage: ./triangle benchmark vertices nnz\n");
}

/**
 * This main function takes as input the number of runs. It reads the data
 * through the input function and then computes the number of triangles.
 * It prints at the end the average execution time for computing the number
 * of triangles given a data set.
 */
int main(int argc, char **argv)
{

	if (argc < 4) {
		help();
		return -1;
	}

	string file_name = argv[1];
	int N = atoi(argv[2]);
	int NUM_A = atoi(argv[3]);

	uint32_t *IA = (uint32_t*)malloc((N + 1) * sizeof(uint32_t));
	uint32_t *JA = (uint32_t*)malloc(NUM_A * sizeof(uint32_t));

	if (fill_input(file_name, IA, JA) < 0 ) {
		free(IA);
		free(JA);
		printf("Error parsing file\n");
		return -1;
	}

	uint64_t total_triangle = 0;
	uint64_t total_triangle_ref = 0;
	double start, end, timeTaken = 0;

	for (int i = 0; i < NUM_ITER; i++) {
		start = currentSeconds();
		total_triangle_ref += count_triangles_orig(IA, JA, N, NUM_A);
		end = currentSeconds();
		timeTaken += end - start;
	}
	
	timeTaken = timeTaken/(double)NUM_ITER;
	printf("ref %lf num_triangles = %lu \n", timeTaken, total_triangle_ref/NUM_ITER);

	timeTaken = 0;
	for (int i = 0; i < NUM_ITER; i++) {
		start = currentSeconds();
		total_triangle += count_triangles(IA, JA, N, NUM_A);
		end = currentSeconds();
		timeTaken += end - start;
	}
	timeTaken = timeTaken/(double)NUM_ITER;
	printf("new %lf num_triangles = %lu \n", timeTaken, total_triangle/NUM_ITER);

	free(IA);
	free(JA);

	if (total_triangle_ref != total_triangle) {
		printf("Correctness FAIL\n");
		return -1;
	}

	return 0;
}
