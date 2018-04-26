CC=g++ -m64
OMP=-fopenmp -DOMP
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61

DEBUG=0
CFLAGS=-g -O3 -Wall -DDEBUG=$(DEBUG) -std=c++11

CFILES = triangle.cpp main.cpp cycletimer.cpp
HFILES = triangle.h cycletimer.h
OBJS = triangle_cuda.o

all: triangle triangle-omp triangle-cuda

triangle: $(CFILES) $(HFILES) 
	$(CC) $(CFLAGS) -o triangle $(CFILES)

triangle-omp: $(CFILES) $(HFILES)
	$(CC) $(CFLAGS) $(OMP) -o triangle-omp $(CFILES)

triangle-cuda: $(CFILES) $(HFILES) $(OBJS)
	$(CC) $(CFLAGS) -DCUDA -o $@ $(OBJS) $(CFILES) $(LDFLAGS)

%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@

clean:
	rm -f *.o
	rm -f triangle triangle-omp triangle-cuda
