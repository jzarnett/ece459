#include <thread>
#include <iostream>

void run() {
  std::cout <<"in run\n";
}

int main() {
  std::thread t1(run);
  std::cout << "in main\n";
  t1.join();
}
