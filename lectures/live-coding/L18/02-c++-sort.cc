/* C++, STL qsort on a vector */
#include <ctime>
#include <cstdlib>

#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

typedef int T;

void test(int N)
{
    if (N < 1) { return; } 

    vector<T> data(N);
    for (int i = 0; i < N; i++)
        data[i] = rand();

   sort(data.begin(), data.end());
}

int main(int argc, char * argv[])
{
    int i;
    if (argc != 2) exit(0);
    for (i = 0; i < 10; ++i)
        test(atoi(argv[1]));
    return 0;
}
