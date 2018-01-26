#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>

#define BUFFER_SIZE 20
#define NUM_PROD 5
#define NUM_CON  3

sem_t spaces;
sem_t items;
int counter;
int* buffer;
int pindex = 0;
int cindex = 0;
int ctotal = 0;
pthread_mutex_t prod_mutex;
pthread_mutex_t con_mutex;

int produce() {
  ++counter;
  printf("Produced %d.\n", counter);
  return counter;
} 

void consume( int value ) {
  printf("Consumed %d.\n", value);
}

void* producer( void* arg ) {
  while( 1 ) {
    pthread_mutex_lock( &prod_mutex );
    if (counter == 10000) {
      pthread_mutex_unlock( &prod_mutex );
      break;
    }
    int v = produce();
    sem_wait( &spaces );
    buffer[pindex] = v;
    pindex = (pindex + 1) % BUFFER_SIZE;
    pthread_mutex_unlock( &prod_mutex );
    sem_post( &items );
  }
  pthread_exit( NULL );
}

void* consumer( void* arg ) {
  while( 1 ) {
    pthread_mutex_lock( &con_mutex ); 
    if (ctotal == 10000) {
      pthread_mutex_unlock( &con_mutex );
      break;
    }
    sem_wait( &items );
    int temp = buffer[cindex];
    buffer[cindex] = -1;
    cindex = (cindex + 1) % BUFFER_SIZE;
    ++ctotal;
    pthread_mutex_unlock( &con_mutex );
    sem_post( &spaces );
    consume( temp );
  }
  pthread_exit( NULL );
}

int main( int argc, char** argv ) {
  counter = 0;
  buffer = malloc( BUFFER_SIZE * sizeof( int ) );
  for ( int i = 0; i < BUFFER_SIZE; i++ ) {
    buffer[i] = -1;
  }  
  sem_init( &spaces, 0, BUFFER_SIZE );
  sem_init( &items, 0, 0 );
  pthread_mutex_init( &prod_mutex, NULL );
  pthread_mutex_init( &con_mutex, NULL );

  pthread_t prod[NUM_PROD];
  pthread_t con[NUM_CON];

  for (int j = 0; j < NUM_PROD; ++j ) {
    pthread_create( &prod[j], NULL, producer, NULL );
  }
  for (int k = 0; k < NUM_CON; ++k ) { 
    pthread_create( &con[k], NULL, consumer, NULL );
  }

  for (int j = 0; j < NUM_PROD; ++j ) {
    pthread_join( prod[j], NULL );
  }
  for (int k = 0; k < NUM_CON; ++k ) {
    pthread_join( con[k], NULL );
  }
  free( buffer );
  sem_destroy( &spaces );
  sem_destroy( &items );
  pthread_mutex_destroy( &prod_mutex );
  pthread_mutex_destroy( &con_mutex );

  pthread_exit(0);
}

