#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void foo( int i, int j ) {
    sleep( 1 );
    printf("i = %d, j=%d.\n", i, j );
}


int main( int argc, char** argv) {

    #pragma omp parallel 
    {
        #pragma omp master
        {
            for ( int i = 0; i < 10; i++ ) {
                #pragma omp task
                {
                    for (int j = 0; j < 20; j++ ) {
                        #pragma omp task
                        foo( i, j );
                    }
                    #pragma omp taskwait 
                }
                if (i == 9) {
                    printf("Finished creating first level tasks.\n");
                }
            }

            #pragma omp taskwait 
            printf("Finished.\n");
        }
    }
    printf("Finished parallel section.\n");
    return 0;
}

