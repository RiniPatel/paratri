/**
 * 15-618 Project
 * @author Tushar Goyal (tgoyal1)
 * @author Rini Patel (rinip)
 */

#include <string>
#include "triangle.h"
#include "cycletimer.h"

using namespace std;

#define NUM_ITER 10
#define DUMP_OUTPUT 0

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

int dump_triangle_to_disk(string file_name, uint32_t * triangle_list, uint32_t num_triangles)
{
#if DUMP_OUTPUT
	FILE * file_triangles = fopen(file_name.c_str(), "w");

	if (file_triangles == NULL) {
		return -1;
	}
	printf("Dumping %d triangles \n", num_triangles);
	for (uint32_t i = 0; i < num_triangles; i++)
	{
		int idx = i * 3;
		fprintf(file_triangles, "%d %d %d\n", triangle_list[idx], triangle_list[idx + 1], triangle_list[idx + 2]);
	}
	fflush(file_triangles);

	fclose(file_triangles);
#endif
	return 0;
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
	uint32_t * triangle_list = (uint32_t*)malloc(MAX_TRIANGLES * 3 * sizeof(uint32_t));

	if (fill_input(file_name, IA, JA) < 0 ) {
		free(IA);
		free(JA);
		printf("Error parsing file\n");
		return -1;
	}

	uint64_t total_triangle_ref = 0;
	double start, end, timeTaken = 0;

	for (int i = 0; i < NUM_ITER; i++) {
		start = currentSeconds();
		total_triangle_ref += count_triangles_orig(IA, JA, N, NUM_A, triangle_list);
		end = currentSeconds();
		timeTaken += end - start;
	}
	dump_triangle_to_disk(file_name + "_ref_triangles.txt", triangle_list, total_triangle_ref / NUM_ITER);
	
	timeTaken = timeTaken/(double)NUM_ITER;
	printf("ref %lf num_triangles = %lu \n", timeTaken, total_triangle_ref/NUM_ITER);

	#if OMP
	uint64_t total_triangle_omp = 0;
	timeTaken = 0;
	for (int i = 0; i < NUM_ITER; i++) {
		start = currentSeconds();
		total_triangle_omp += count_triangles_omp(IA, JA, N, NUM_A, triangle_list);
		end = currentSeconds();
		timeTaken += end - start;
	}

	dump_triangle_to_disk(file_name + "_omp_triangles.txt", triangle_list, total_triangle_omp / NUM_ITER);

	timeTaken = timeTaken/(double)NUM_ITER;
	printf("new %lf num_triangles = %lu \n", timeTaken, total_triangle_omp/NUM_ITER);
	#endif

	#ifdef CUDA
	timeTaken = 0;
	uint64_t total_triangle_cuda = 0;
	for (int i = 0; i < NUM_ITER; i++) {
		start = currentSeconds();
		total_triangle_cuda += count_triangles_cuda(IA, JA, N, NUM_A, triangle_list);
		end = currentSeconds();
		timeTaken += end - start;
	}

	dump_triangle_to_disk(file_name + "_cuda_triangles.txt", triangle_list, total_triangle_cuda / NUM_ITER);

	timeTaken = timeTaken/(double)NUM_ITER;
	printf("cuda %lf num_triangles = %lu \n", timeTaken, total_triangle_cuda/NUM_ITER);
	#endif

	free(IA);
	free(JA);

	#if OMP
	if (total_triangle_ref != total_triangle_omp)
	#endif
	#if CUDA
	if (total_triangle_ref != total_triangle_cuda)
	#endif
	{
		printf("Correctness FAIL\n");
		return -1;
	}

	return 0;
}
