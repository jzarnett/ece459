// ECE 459 Live Coding example
// Lecture 12
// Can gcc parallelize it?
//   gcc L12-can-gcc-autoparallelize-it.c -O2 -floop-parallelize-all -ftree-parallelize-loops=4 -fopt-info -DLn

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main() {
  int i, j;
  int x[1000];
  int X[100][100], Y[100][100];

  #ifdef L1
  for (i = 0; i < 1000; i++)
      x[i] = i + 3;
  #endif

  #ifdef L2
  for (i = 0; i < 100; i++)
    for (j = 0; j < 100; j++)
      X[i][j] = X[i][j] + Y[i-1][j];
  #endif

  #ifdef L3
  for (i = 0; i < 10; i++)
    x[2*i+1] = x[2*i];
  #endif

  #ifdef L4
  for (j = 0; j <= 10; j++)
    if (j > 5) x[i] = i + 3;
  #endif

  #ifdef L5
  for (i = 0; i < 100; i++)
    for (j = i; j < 100; j++)
      X[i][j] = 5;
  #endif
  
  printf("x[0] = %d\n", x[0]);
  printf("X[0][0] = %d\n", X[0][0]);
}

