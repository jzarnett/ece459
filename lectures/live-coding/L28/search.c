// reminder: compile with -pthread
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define ARRAY_SIZE 30
#define NUM_THREADS 8
int array[ARRAY_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 27, 10, 18, 199, 8585, 18, 15, 12, 14, 56, 78, 99, 108, 48, 49};


typedef struct parameter {
    int startIndex;
    int endIndex;
    int searchValue;
} parameter_t;

void *search( void *void_arg );

int main( int argc, char** argv ) {
    int search_val;
    
    if ( argc < 2 ) {
        printf("A search value is required.\n");
       return -1;
    }
    search_val = atoi( argv[1] );

    pthread_t threads[NUM_THREADS];
    void* returnValue;
    for ( int i = 0; i < NUM_THREADS; ++i ) {
        parameter_t* params = malloc( sizeof ( parameter_t ) );
        params->startIndex = i * (ARRAY_SIZE / NUM_THREADS);
        int end = (i + 1) * (ARRAY_SIZE / NUM_THREADS);
        if ( i == (NUM_THREADS - 1) ) {
            end = ARRAY_SIZE;
        }
        params->endIndex = end;
        params->searchValue = search_val;

        pthread_create(&threads[i], NULL, search, params);
    }
    
    for ( int i = 0; i < NUM_THREADS; ++i ) {
        pthread_join( threads[i], &returnValue );
        int *rv = (int *) returnValue;
        if ( -1 != *rv ) {
            printf("Found at %d by thread %d\n", *rv, i);
        }
        free( returnValue );
    }

    pthread_exit(0);
}

void *search( void *void_arg ) {
    parameter_t *arg = (parameter_t *) void_arg;
    int *result = malloc( sizeof( int ) );
    *result = -1; // Default value

    for ( int i = arg->startIndex; i < arg->endIndex; ++i ) {
        if ( array[i] == arg->searchValue ) {
            *result = i;
            break;
        }
    }
    free( void_arg );
    pthread_exit(result);
}

