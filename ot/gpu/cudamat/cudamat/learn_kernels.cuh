#ifndef EBM_KERNELS_H_
#define EBM_KERNELS_H_

#define NUM_VECTOR_OP_BLOCKS                4096
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     512

#define NUM_SPARSE_GRAD_BLOCKS               4096
#define NUM_SPARSE_GRAD_THREADS_PER_BLOCK    512

__global__ void kMultiplyBySigmoidGrad(double* act, double* target, const unsigned int len);

#endif
