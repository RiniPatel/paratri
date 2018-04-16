run_small:
	gcc -O2 triangle.c -o triangle.x -std=c99 -fopenmp
	cat graphs/small_IA.txt graphs/small_JA.txt | ./triangle.x 6474 25144

run_medium:
	gcc -O2 triangle.c -o triangle.x -std=c99 -fopenmp
	cat graphs/medium_IA.txt graphs/medium_JA.txt | ./triangle.x 9877 51946

run_large:
	gcc -O2 triangle.c -o triangle.x -std=c99 -fopenmp
	cat graphs/large_IA.txt graphs/large_JA.txt | ./triangle.x 22687 109410

clean:
	rm -rf *~
	rm -rf *.x

