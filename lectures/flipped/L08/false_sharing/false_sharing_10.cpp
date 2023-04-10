#include <iostream>
#include <boost/thread.hpp>

class threadS {
public:
   threadS(unsigned char *aVector, unsigned int aVSize)
   :theVector(aVector),
    theVSize(aVSize)
   { }

   void operator()() {
      unsigned long long myCounter = 100000000;
      while(--myCounter) {
          for (int i=0; i<10; ++i) {
              ++theVector[i];
          }
      }
   }
private:
   unsigned char* theVector;
   unsigned int   theVSize;
};

int main()
{
   unsigned char vectorA[10];
   unsigned char vectorB[10];

   // unsigned char vectorA[100];
   // unsigned char vectorB[100];

   std::cout << std::hex;
   std::cout << "A:[" <<  (long)&vectorA[0] << "-" << (long)&vectorA[9] << "]" << std::endl;
   std::cout << "B:[" <<  (long)&vectorB[0] << "-" << (long)&vectorB[9] << "]" << std::endl;

   threadS threadA(vectorA, 10);
   threadS threadB(vectorB, 10);

   boost::thread_group tg;
   tg.create_thread(threadA);
   tg.create_thread(threadB);

   tg.join_all();
}
