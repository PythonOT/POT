#include "learn_kernels.cuh"

__global__ void kMultiplyBySigmoidGrad(double* act, double* target, const unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for(unsigned int i = idx; i < len; i+= numThreads) {
        target[i] = target[i] * act[i] * (1.0f - act[i]);
    }
}
