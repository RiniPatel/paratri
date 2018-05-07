CC=g++ -m64
OMP=-fopenmp -DOMP
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart -lcudadevrt
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -rdc=true

DEBUG=0
CFLAGS=-g -O3 -Wall -DDEBUG=$(DEBUG) -std=c++11

CFILES = triangle.cpp main.cpp cycletimer.cpp
HFILES = triangle.h cycletimer.h
OBJS = triangle_cuda.o triangle_cuda_link.o

all: triangle triangle-omp triangle-cuda

triangle: $(CFILES) $(HFILES) 
	$(CC) $(CFLAGS) -o triangle $(CFILES)

triangle-omp: $(CFILES) $(HFILES)
	$(CC) $(CFLAGS) $(OMP) -o triangle-omp $(CFILES)

triangle-cuda: $(CFILES) $(HFILES) $(OBJS)
	$(CC) $(CFLAGS) -DCUDA -o $@ $(OBJS) $(CFILES) $(LDFLAGS)

%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@
	$(NVCC) -arch=compute_61 -dlink -o triangle_cuda_link.o triangle_cuda.o -lcudadevrt -lcudart

clean:
	rm -f *.o
	rm -f triangle triangle-omp triangle-cuda
