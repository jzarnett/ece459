#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


int main( int argc, char** argv ) {
  #pragma omp parallel 
  {
   #pragma omp master 
   {
     printf("Today's threads brought to you by thread %d.\n", omp_get_thread_num());
     for ( int i = 0; i < 10; i++ ) {
      #pragma omp task 
      {
        printf("Created child task %d.\n", i);
        for (int j = 0; j < 10; j++ ) {
           #pragma omp task 
           {
               sleep( 5 );
           }
        }
      }
     }
    #pragma omp taskwait
    printf("Waited for all children.\n");
   }
  }
  printf("Ready to return.\n");
  return 0;
}
