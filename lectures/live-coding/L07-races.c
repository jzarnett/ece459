#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void * run1(void *arg) {
    int * x = (int *) arg;
    sleep (0L);
    for (int i = 0; i < 1000; i++) {
      *x += 1;
    }
}

void * run2 (void * arg) {
    int * x = (int *) arg;
    sleep (0L);
    for (int i = 0; i < 1000; i++) {
      *x += 2;
    }
}

int main(int argc, char * argv[]) {
    int *x = malloc(sizeof(int));
    *x = 1;
    pthread_t t1, t2;
    pthread_create(&t1, NULL, &run1, x);
    pthread_create(&t2, NULL, &run2, x);
    pthread_join(t1, NULL);
    pthread_join(t2,NULL);
    printf("%d\n", *x);
    free(x);
    return EXIT_SUCCESS;
}

