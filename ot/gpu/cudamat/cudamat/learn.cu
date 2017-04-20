#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include "learn_kernels.cuh"
#include "cudamat.cuh"

extern "C" {

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

EXPORT int mult_by_sigmoid_deriv(cudamat* target, cudamat* acts) {
    int len = acts->size[0]*acts->size[1];

    if (acts->is_trans != target->is_trans)
        return ERROR_TRANSPOSED;

    if (acts->size[0] != target->size[0] || acts->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultiplyBySigmoidGrad<<<NUM_VECTOR_OP_BLOCKS,NUM_VECTOR_OP_THREADS_PER_BLOCK>>>(acts->data_device, target->data_device, len);

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

}
