CC=g++
OMP=-fopenmp -DOMP

DEBUG=0
CFLAGS=-g -O3 -Wall -DDEBUG=$(DEBUG) -std=c++11
LDFLAGS= -lm

CFILES = triangle.cpp triangle_ref.cpp cycletimer.cpp
HFILES = triangle_ref.h cycletimer.h

GFILES = gengraph.py grun.py rutil.py sim.py viz.py  regress.py benchmark.py grade.py

all: triangle triangle-omp

triangle: $(CFILES) $(HFILES) 
	$(CC) $(CFLAGS) -o triangle $(CFILES) $(LDFLAGS)

triangle-omp: $(CFILES) $(HFILES)
	$(CC) $(CFLAGS) $(OMP) -o triangle-omp $(CFILES) $(LDFLAGS)

clean:
	rm -f *.o
	rm -f triangle triangle-omp
