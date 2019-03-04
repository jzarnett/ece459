/* C++, STL qsort on an array */
#include <ctime>
#include <cstdlib>

#include <iostream>
#include <algorithm>

using namespace std;

typedef int T;

void test(int N)
{
    T* data;

    if (N < 1) { return; } 

    data = new T[N];
    if (!data) return;
    for (int i = 0; i < N; i++)
        data[i] = rand();

    time_t begin = clock();
    sort(data, data+N);

    time_t end = clock();
    double time_elapsed = (double) (end-begin) / CLOCKS_PER_SEC;

    cout << time_elapsed << endl;
    delete[] data;
}

int main(int argc, char * argv[])
{
    int i;
    if (argc != 2) exit(0);
    for (i = 0; i < 10; ++i)
        test(atoi(argv[1]));
    return 0;
}
