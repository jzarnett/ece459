01-qsort: 01-qsort.c
	gcc -O2 -o 01-qsort 01-qsort.c
	perf stat -r 5 ./01-qsort 1000000

02-c++-sort: 02-c++-sort.cc
	g++ -O2 -o 02-c++-sort 02-c++-sort.cc
	perf stat -r 5 ./02-c++-sort 1000000

03-custom-sort: 03-custom-sort.c
	gcc -O2 -o 03-custom-sort 03-custom-sort.c
	perf stat -r 5 ./03-custom-sort 1000000

04-vector-vs-list: 04-vector-vs-list.cc
	g++ -std=c++11 -O2 -o 04-vector-vs-list 04-vector-vs-list.cc
	./04-vector-vs-list 100000

