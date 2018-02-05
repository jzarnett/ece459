  double calc(int count) {
    double d = 1.0;
    for (int i = 0; i < count*count; i++) d += d;
    return d;
  }

  int main() {
    double data[200][200];
    int i, j;
    #pragma omp parallel for private(i, j) shared(data)
    for (int i = 0; i < 200; i++) {
      for (int j = 0; j < 200; j++) {
        data[i][j] = calc(i+j);
      }
    }
    return 0;
  }
