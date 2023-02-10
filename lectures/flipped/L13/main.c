#include <stdio.h>
#include <float.h>
#include <math.h>

int float_to_int(float f) {
  return *(int*)&f;
}

int main() {
    float f = 0.01;
    printf("float,int\n");
    while (f < 10) {
        printf("%.4f,%d\n", f, float_to_int(f));
        f += 0.0001;
    }
    return 0;
}
