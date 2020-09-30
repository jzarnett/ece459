#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main( int argc, char** argv) {
   #pragma omp parallel
   {
     #pragma omp single 
     {
         printf("In Single Section.\n");
         for( int i = 0; i < 10; i++ ) {
         #pragma omp task
         {
           printf("Task created.\n");
      
           for( int j = 0; j < 10; ++j ) {
             #pragma omp task 
             {
                sleep(5);
             }
           }
         }
       }
   
       #pragma omp taskwait
       printf("Waited for all children.\n");
     }
     printf("Outside of Single Section.\n");
  }

  return 0;
}

