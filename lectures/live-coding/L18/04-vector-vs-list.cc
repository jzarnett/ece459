#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <list>
#include <vector>
#include <iterator>
#include <algorithm>

using namespace std;

// Slightly unsafe, should have a static assert that C<T>, T actually holds
template <class C, class T> void test(const unique_ptr<T[]>& array,
                                      const unique_ptr<T[]>& position,
                                      const T& N)
{
    // Create the container
    C container;

    auto start = clock();

    // Inserting elements into sequence
    for (T i = 0; i < N; ++i) {
        auto iterator = lower_bound(container.begin(), container.end(), array[i]);
        container.insert(iterator, array[i]);
    }

    auto end = clock();
    double insert_time = (end - start) / ((double) CLOCKS_PER_SEC);
    cout << "insert " << insert_time << "s";
    start = clock();

    // Removing the elements one at a time, based on random positions
    for (T i = 0; i < N; ++i) {
        auto iterator = container.begin();
        advance(iterator, position[i]);
        container.erase(iterator);
    }

    end = clock();
    double remove_time = (end - start) / ((double) CLOCKS_PER_SEC);
    cout << "   remove " << remove_time << "s   total "
         << insert_time + remove_time << "s" << endl;
}

int main(int argc, char *argv[])
{
    // Setup
    if (argc != 2) {
        abort();
    }
    int N = atoi(argv[1]);
    if (N <= 0) {
        abort();
    }
    srand(time(NULL));

    for (int j = 1; j <= 6; ++j) {
        // Array of random integers
        unique_ptr<int[]> a(new int [N]);
        for (int i = 0; i < N; ++i) {
            a[i] = rand();
        }

        // Array of random positions
        unique_ptr<int[]> p(new int[N]);
        for (int i = 0; i < N; ++i) {
            p[i] = rand() % (N - i);
        }

        // Do the tests
        cout << "Test " << j << endl;
        cout << "======" << endl;
        cout << "vector: ";
        test <vector<int>> (a, p, N);
        cout << "list:   ";
        test <list<int>, int> (a, p, N);
    }
    return 0;
}
