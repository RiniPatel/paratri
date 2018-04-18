all:
	g++ -g -O2 triangle.cpp -o triangle -fopenmp

clean:
	rm -rf *~
	rm -rf *.x

