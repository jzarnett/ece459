extern "C" __global__ void vector_add(float* A, float* B, float* C)
{
  int i = blockIdx.x;
  C[i] = A[i] + B[i];
}
