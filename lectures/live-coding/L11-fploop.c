// ECE 459 Live Coding Example
// Lecture 11: auto-parallelizable version of matrix/vector multiply
//  $ /opt/oracle/solarisstudio12.3/bin/cc fploop.c -c -xautopar -xloopinfo


void matVec(double **mat, double *vec, double * restrict out, int * restrict row, int * restrict col) {
    int i, j;
    for (i = 0; i < *row; i++) {
      out[i] = 0;
      for (j = 0; j < *col; j++) {
        out[i] += mat[i][j] * vec[j];
      }
    }
}
