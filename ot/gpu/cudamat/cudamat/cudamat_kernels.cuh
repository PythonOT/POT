#ifndef NVMATRIX_KERNEL_H_
#define NVMATRIX_KERNEL_H_

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
 * Defines for getting the values at the lower and upper 32 bits
 * of a 64-bit number.
 */
#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

/*
 * Number of iterations to run random number generator upon initialization.
 */
#define NUM_RND_BURNIN                      100

/*
 * CUDA grid dimensions for different types of kernels
 */
#define COPY_BLOCK_SIZE                     16
#
// element-wise kernels use min(ceil(N / 512), 4096) blocks of 512 threads
#define MAX_VECTOR_OP_BLOCKS                4096
#define MAX_VECTOR_OP_THREADS_PER_BLOCK     512
#define NUM_VECTOR_OP_BLOCKS(N)             (min(((N) + MAX_VECTOR_OP_THREADS_PER_BLOCK - 1)/MAX_VECTOR_OP_THREADS_PER_BLOCK, MAX_VECTOR_OP_BLOCKS))
#define NUM_VECTOR_OP_THREADS_PER_BLOCK(N)  (min((N), MAX_VECTOR_OP_THREADS_PER_BLOCK))

#define PI 3.1415926535897932f

__global__ void kSeedRandom(unsigned int* randMults, unsigned long long* randWords, unsigned int seed);
__global__ void kRandomUniform(unsigned int* randMults, unsigned long long* randWords, double* gData, unsigned int numElements);
__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, double* gData, unsigned int numElements);

__global__ void kGetRowSlice(double* source, double* target, int start, int end, int width, int height);
__global__ void kTranspose(double *odata, double *idata, int width, int height);
__global__ void kSetRowSlice(double* source, double* target, int start, int end, int width, int height);

__global__ void kLessThan(double* mat1, double* mat2, double* target, unsigned int len);
__global__ void kLessThanScalar(double* mat, double val, double* target, unsigned int len);
__global__ void kGreaterThan(double* mat1, double* mat2, double* target, unsigned int len);
__global__ void kGreaterThanScalar(double* mat, double val, double* target, unsigned int len);
__global__ void kEquals(double* mat1, double* mat2, double* target, unsigned int len);
__global__ void kEqualsScalar(double* mat, double val, double* target, unsigned int len);
__global__ void kMinimum(double* mat1, double* mat2, double* target, unsigned int len);
__global__ void kMinimumScalar(double* mat, double val, double* target, unsigned int len);
__global__ void kMaximum(double* mat1, double* mat2, double* target, unsigned int len);
__global__ void kMaximumScalar(double* mat, double val, double* target, unsigned int len);
__global__ void kMinColumnwise(double* mat, double* target, unsigned int width, unsigned int height);
__global__ void kMinRowwise(double* mat, double* target, unsigned int width, unsigned int height);
__global__ void kMaxColumnwise(double* mat, double* target, unsigned int width, unsigned int height);
__global__ void kMaxRowwise(double* mat, double* target, unsigned int width, unsigned int height);
__global__ void kArgMinColumnwise(double* mat, double* target, unsigned int width, unsigned int height);
__global__ void kArgMinRowwise(double* mat, double* target, unsigned int width, unsigned int height);
__global__ void kArgMaxColumnwise(double* mat, double* target, unsigned int width, unsigned int height);
__global__ void kArgMaxRowwise(double* mat, double* target, unsigned int width, unsigned int height);
__global__ void kSign(double* mat, double* target, unsigned int len);
__global__ void kApplySigmoid(double* mat, double* target, unsigned int len);
__global__ void kApplyTanh(double* mat, double* target, unsigned int len);
__global__ void kApplySoftThreshold(double* mat, double alpha, double* target, unsigned int len);
__global__ void kApplyAbs(double* mat, double* target, unsigned int len);
__global__ void kApplyLog1PlusExp(double* mat, double* target, unsigned int len);
__global__ void kLog(double* mat, double* target, unsigned int len);
__global__ void kExp(double* mat, double* target, unsigned int len);
__global__ void kGamma(double* mat, double* target, unsigned int len);
__global__ void kLogGamma(double* mat, double* target, unsigned int len);
__global__ void kSqrt(double* mat, double* target, unsigned int len);
__global__ void kPow(double* mat, double pow, double* target, unsigned int len);
__global__ void kPowMatrix(double* mat, double* pow, double* target, unsigned int len);
__global__ void kReciprocal(double* mat, double* target, unsigned int len);
__global__ void kAddColVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height);
__global__ void kAddRowVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height);
__global__ void kAddColMult(double* mat, double* vec, double* tgtMat, double mult, unsigned int width, unsigned int height);
__global__ void kMultByColVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height);
__global__ void kMultByRowVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height);
__global__ void kDivByColVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height);
__global__ void kDivByRowVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height);
__global__ void kAdd(double* a, double* b, double* dest, unsigned int numEls);
__global__ void kSubtract(double* a, double* b, double* dest, unsigned int numEls);
__global__ void kMult(double* a, double* b, double* dest, unsigned int numEls);
__global__ void kDivide(double* a, double* b, double* dest, unsigned int numEls);
__global__ void kMultScalar(double* mat, double alpha, double* dest, unsigned int len);
__global__ void kAssignScalar(double* dest, double alpha, unsigned int len);
__global__ void kDivideScalar(double* mat, double alpha, double* dest, unsigned int len);
__global__ void kAddScalar(double* a, double alpha, double* dest, unsigned int numEls);
__global__ void kSelectRows(double* source, double* target, double* indices, int nRowIs, int nCols, int nSourceRows);
__global__ void kSetSelectedRows(double* target, double* source, double* indices, int nRowIs, int nCols, int nTargetRows);
__global__ void kWhere(double* condition_mat, double* if_mat, double* else_mat, double* target, unsigned int len);
__global__ void kCorrelate(double* source, double* kernel, double* dest, int width, int height, int fwidth, int fheight);
#endif
