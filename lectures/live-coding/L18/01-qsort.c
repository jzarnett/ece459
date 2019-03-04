/* C, library qsort */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

typedef int T;

int compare_T (const void *a, const void *b) 
{
    return (*(int *)a - *(int *)b);
}

void test(int N)
{
    T* data;
    time_t begin, end;
    double time_elapsed;
    int i;

    if (N < 1) { printf("N must be at least 1\n"); return; }

    data = (T*) malloc(N*sizeof(T));
    if (!data) return;
    for (i = 0; i < N; i++)
        data[i] = rand();

    qsort(data, N, sizeof(T), compare_T);

    free(data);
}

int main(int argc, char * argv[])
{
    int i;
    if (argc != 2) exit(0);
    for (i = 0; i < 10; ++i)
        test(atoi(argv[1]));
    return 0;
}
