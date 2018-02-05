// ECE 459 Live Coding example
// Lecture 11
// Un-parallelized vector processing example;
//  can be autoparallelized by Solaris compiler
//  $ /opt/oracle/solarisstudio12.3/bin/cc omp_vector.c -o omp_vector -xautopar


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void setup(double * vector, int length) {
  int i;
  for (i = 0; i < length; i++) {
    vector[i] += 1.0;
  }
}

int main() {
  double * vector;
  vector = (double *) malloc(sizeof(double) * 1024 * 1024);
  memset(vector, 0, sizeof(double) * 1024 * 1024);
  for (int i = 0; i < 1000; i++) {
    setup(vector, 1024*1024);
  }
  printf("vector[0] = %f\n", vector[0]);
}

