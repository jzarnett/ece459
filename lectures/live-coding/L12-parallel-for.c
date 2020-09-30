  void calc (double *array1, double *array2, int length) {
    #pragma omp parallel for
    for (int i = 0; i < length; i++) {
      array1[i] += array2[i];
    }
  }
