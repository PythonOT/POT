#include "cudamat_kernels.cuh"
#include "float.h"

/* ------------------------- Random number generation ------------------------- */

__global__ void kSeedRandom(unsigned int* rndMults, unsigned long long* rndWords, unsigned int seed) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // The initial x is the seed and the initial carry is 1
    unsigned long long rndWord = ((unsigned long long)seed << 32) + 1;
    const unsigned int rndMult = rndMults[idx];
    /*
     * Run the chain for a few steps so that all the streams have a chance
     * to differentiate. They start out generating similar random numbers
     * because all the multipliers are similar.
     */
    for(unsigned int i = 0; i < NUM_RND_BURNIN; i++) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
    }
    rndWords[idx] = rndWord;
}

__global__ void kRandomUniform(unsigned int* rndMults, unsigned long long* rndWords, double* gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    for(unsigned int i = idx; i < numElements; i += NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        gData[i] = (__uint2double_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
    }
    rndWords[idx] = rndWord;
}

__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, double* gData, unsigned int numElements) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long rndWord = rndWords[idx];
    const unsigned int rndMult = rndMults[idx];

    double rnd1, rnd2, R, T;
    for(unsigned int i = idx; i < numElements; i += 2*NUM_RND_STREAMS) {
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd1 = (__uint2double_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        rndWord = rndMult * LOW_BITS(rndWord) + HIGH_BITS(rndWord);
        rnd2 = (__uint2double_rn(LOW_BITS(rndWord)) + 1.0f) / 4294967296.0f;
        T = 2 * PI * rnd2;
        R = sqrtf(-2 * __logf(rnd1));
        gData[i] = R * __cosf(T);
        if (i + NUM_RND_STREAMS < numElements)
            gData[i + NUM_RND_STREAMS] = R * __sinf(T);
    }
    rndWords[idx] = rndWord;
}

/* ------------------------- Data copying ------------------------- */

/*
Copy row slice from source to target. There is a block for every 32x32 chunk being copied.
*/
__global__ void kGetRowSlice(double* source, double* target, int start, int end, int width, int height) {
    const int row = start + blockIdx.x * 32 + threadIdx.x;
    const int start_col = blockIdx.y * 32;

    const int end_col = (start_col + 32 < width) ? start_col + 32: width;

    const int target_height = end - start;

    if (row < end) {
        for (int cur_col = start_col; cur_col < end_col; cur_col++)
            target[cur_col * target_height + row - start] = source[cur_col * height + row];
    }
}

__global__ void kSetRowSlice(double* source, double* target, int start, int end, int width, int height) {
    const int row = start + blockIdx.x * 32 + threadIdx.x;
    const int start_col = blockIdx.y * 32;

    const int end_col = (start_col + 32 < width) ? start_col + 32: width;

    const int source_height = end - start;

    if (row < end) {
        for (int cur_col = start_col; cur_col < end_col; cur_col++)
            target[cur_col * height + row] = source[cur_col * source_height + row - start];
            //source[cur_col * height + row - start] = target[cur_col * target_height + row];
    }
}

__global__ void kTranspose(double *odata, double *idata, int width, int height) {
    __shared__ double block[COPY_BLOCK_SIZE][COPY_BLOCK_SIZE+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.x;
    unsigned int yIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < width) && (yIndex < height)) {
        unsigned int index_in = yIndex * width + xIndex;

        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * COPY_BLOCK_SIZE + threadIdx.x;
    yIndex = blockIdx.x * COPY_BLOCK_SIZE + threadIdx.y;

    if((xIndex < height) && (yIndex < width)) {
        unsigned int index_out = yIndex * height + xIndex;

        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

/* ------------------------- Mathematical operations ------------------------- */

__global__ void kLessThan(double* mat1, double* mat2, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat1[i] < mat2[i];
    }
}

__global__ void kLessThanScalar(double* mat, double val, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] < val;
    }
}

__global__ void kGreaterThan(double* mat1, double* mat2, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat1[i] > mat2[i];
    }
}

__global__ void kGreaterThanScalar(double* mat, double val, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] > val;
    }
}

__global__ void kEquals(double* mat1, double* mat2, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat1[i] == mat2[i];
    }
}

__global__ void kEqualsScalar(double* mat, double val, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] == val;
    }
}

__global__ void kMinimum(double* mat1, double* mat2, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = fminf(mat1[i], mat2[i]);
    }
}

__global__ void kMinimumScalar(double* mat, double val, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = fminf(mat[i], val);
    }
}

__global__ void kMaximum(double* mat1, double* mat2, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = fmaxf(mat1[i], mat2[i]);
    }
}

__global__ void kMaximumScalar(double* mat, double val, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = fmaxf(mat[i], val);
    }
}

__global__ void kMinColumnwise(double* mat, double* target, unsigned int width, unsigned int height) {
    __shared__ double min_vals[32];
    double cur_min = FLT_MAX;
    double val = 0;
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];

        if (val < cur_min)
            cur_min = val;
    }

    min_vals[threadIdx.x] = cur_min;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_min = FLT_MAX;

        for (unsigned int i = 0; i < 32; i++)
            if (min_vals[i] < cur_min)
                cur_min = min_vals[i];

        target[blockIdx.x] = cur_min;
    }
}

__global__ void kMinRowwise(double* mat, double* target, unsigned int width, unsigned int height) {
    __shared__ double min_vals[32];
    double cur_min = FLT_MAX;
    double val = 0;
 
    for (unsigned int i = threadIdx.x; i < width; i += 32) {
        val = mat[i * height + blockIdx.x];

        if (val < cur_min)
            cur_min = val;
    }

    min_vals[threadIdx.x] = cur_min;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_min = FLT_MAX;

        for (unsigned int i = 0; i < 32; i++)
            if (min_vals[i] < cur_min)
                cur_min = min_vals[i];

        target[blockIdx.x] = cur_min;
    }
}

__global__ void kMaxColumnwise(double* mat, double* target, unsigned int width, unsigned int height) {
    __shared__ double max_vals[32];
    double cur_max = -FLT_MAX;
    double val = 0;
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];

        if (val > cur_max)
            cur_max = val;
    }

    max_vals[threadIdx.x] = cur_max;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;

        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max)
                cur_max = max_vals[i];

        target[blockIdx.x] = cur_max;
    }
}

__global__ void kMaxRowwise(double* mat, double* target, unsigned int width, unsigned int height) {
    __shared__ double max_vals[32];
    double cur_max = -FLT_MAX;
    double val = 0;
 
    for (unsigned int i = threadIdx.x; i < width; i += 32) {
        val = mat[i * height + blockIdx.x];

        if (val > cur_max)
            cur_max = val;
    }

    max_vals[threadIdx.x] = cur_max;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;

        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max)
                cur_max = max_vals[i];

        target[blockIdx.x] = cur_max;
    }
}

__global__ void kArgMinColumnwise(double* mat, double* target, unsigned int width, unsigned int height) {
    __shared__ double min_vals[32];
    __shared__ unsigned int min_args[32];
    double cur_min = FLT_MAX;
    unsigned int cur_arg = 0;
    double val = 0;
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];

        if (val < cur_min) {
            cur_min = val;
            cur_arg = i;
        }
    }

    min_vals[threadIdx.x] = cur_min;
    min_args[threadIdx.x] = cur_arg;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_min = FLT_MAX;
        cur_arg = 0;

        for (unsigned int i = 0; i < 32; i++)
            if (min_vals[i] < cur_min) {
                cur_min = min_vals[i];
                cur_arg = min_args[i];
            }

        target[blockIdx.x] = cur_arg;
    }
}

__global__ void kArgMinRowwise(double* mat, double* target, unsigned int width, unsigned int height) {
    __shared__ double min_vals[32];
    __shared__ unsigned int min_args[32];
    double cur_min = FLT_MAX;
    unsigned int cur_arg = 0;
    double val = 0;
 
    for (unsigned int i = threadIdx.x; i < width; i += 32) {
        val = mat[i * height + blockIdx.x];

        if (val < cur_min) {
            cur_min = val;
            cur_arg = i;
        }
    }

    min_vals[threadIdx.x] = cur_min;
    min_args[threadIdx.x] = cur_arg;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_min = FLT_MAX;
        cur_arg = 0;

        for (unsigned int i = 0; i < 32; i++)
            if (min_vals[i] < cur_min) {
                cur_min = min_vals[i];
                cur_arg = min_args[i];
            }

        target[blockIdx.x] = cur_arg;
    }
}

__global__ void kArgMaxColumnwise(double* mat, double* target, unsigned int width, unsigned int height) {
    __shared__ double max_vals[32];
    __shared__ unsigned int max_args[32];
    double cur_max = -FLT_MAX;
    unsigned int cur_arg = 0;
    double val = 0;
 
    for (unsigned int i = threadIdx.x; i < height; i += 32) {
        val = mat[blockIdx.x * height + i];

        if (val > cur_max) {
            cur_max = val;
            cur_arg = i;
        }
    }

    max_vals[threadIdx.x] = cur_max;
    max_args[threadIdx.x] = cur_arg;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;
        cur_arg = 0;

        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max) {
                cur_max = max_vals[i];
                cur_arg = max_args[i];
            }

        target[blockIdx.x] = cur_arg;
    }
}

__global__ void kArgMaxRowwise(double* mat, double* target, unsigned int width, unsigned int height) {
    __shared__ double max_vals[32];
    __shared__ unsigned int max_args[32];
    double cur_max = -FLT_MAX;
    unsigned int cur_arg = 0;
    double val = 0;
 
    for (unsigned int i = threadIdx.x; i < width; i += 32) {
        val = mat[i * height + blockIdx.x];

        if (val > cur_max) {
            cur_max = val;
            cur_arg = i;
        }
    }

    max_vals[threadIdx.x] = cur_max;
    max_args[threadIdx.x] = cur_arg;

    __syncthreads();

    if (threadIdx.x == 0) {
        cur_max = -FLT_MAX;
        cur_arg = 0;

        for (unsigned int i = 0; i < 32; i++)
            if (max_vals[i] > cur_max) {
                cur_max = max_vals[i];
                cur_arg = max_args[i];
            }

        target[blockIdx.x] = cur_arg;
    }
}

__global__ void kSign(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] ? copysignf(1., mat[i]) : 0.;
    }
}

__global__ void kApplySigmoid(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = 1 / (1 + __expf(-mat[i]));
    }
}


__global__ void kApplyTanh(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    double mat_i, exp2x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        mat_i = mat[i];
        exp2x = __expf(2 * mat_i);
        target[i] = 1 - 2 / (exp2x + 1);
    }
}

__global__ void kApplySoftThreshold(double* mat, double alpha, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        double f = mat[i];
        target[i] = f > 0 ? max(0., f - alpha) : min(0., f + alpha);
    }
}

__global__ void kApplyAbs(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = mat[i] * ((mat[i] > 0) - (mat[i] < 0));
    }
}

__global__ void kApplyLog1PlusExp(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    double mat_i;

    for (unsigned int i = idx; i < len; i += numThreads) {
        mat_i = mat[i];
        if (mat_i > 0)
            target[i] = (__logf(1 + __expf(-mat_i)) + mat_i);
        else
            target[i] = __logf(1 + __expf(mat_i));
    }
}

__global__ void kLog(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = __logf(mat[i]);
    }
}

__global__ void kExp(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = __expf(mat[i]);
    }
}

__global__ void kGamma(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = tgammaf(mat[i]);
    }
}

__global__ void kLogGamma(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = lgammaf(mat[i]);
    }
}

__global__ void kSqrt(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = sqrt(mat[i]);
    }
}

__global__ void kPow(double* mat, double pow, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = powf(mat[i], pow);
    }
}

__global__ void kPowMatrix(double* mat, double* pow, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = powf(mat[i], pow[i]);
    }
}

__global__ void kReciprocal(double* mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads)
        target[i] = 1.f / mat[i];
}

__global__ void kAddColVector(double* mat, double* vec, double* tgtMat, unsigned int width,
                              unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + vec[i % height];
    }
}

__global__ void kAddRowVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + vec[i / height];
    }
}

__global__ void kAddColMult(double* mat, double* vec, double* tgtMat, double mult,
                            unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] + mult * vec[i % height];
    }
}

__global__ void kMultByColVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] * vec[i % height];
    }
}

__global__ void kMultByRowVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] * vec[i / height];
    }
}

__global__ void kDivByColVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] / vec[i % height];
    }
}

__global__ void kDivByRowVector(double* mat, double* vec, double* tgtMat, unsigned int width, unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        tgtMat[i] = mat[i] / vec[i / height];
    }
}

__global__ void kAdd(double* a, double* b, double* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] + b[i];
    }
}

__global__ void kSubtract(double* a, double* b, double* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] - b[i];
    }
}

__global__ void kDivide(double* a, double* b, double* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] / b[i];
    }
}

__global__ void kMult(double* a, double* b, double* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] * b[i];
    }
}

__global__ void kMultScalar(double* mat, double alpha, double* dest, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        dest[i] = alpha * mat[i];
    }
}

__global__ void kAssignScalar(double* dest, double alpha, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        dest[i] = alpha;
    }
}

__global__ void kDivideScalar(double* mat, double alpha, double* dest, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < len; i += numThreads) {
        dest[i] = mat[i] / alpha;
    }
}

__global__ void kAddScalar(double* a, double alpha, double* dest, unsigned int numEls) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < numEls; i += numThreads) {
        dest[i] = a[i] + alpha;
    }
}

__global__ void kSelectRows(double* source, double* target, double* indices, int nRowIs, int nCols, int nSourceRows){
    __shared__ int sourceRowIndices[32];
    const int startTargetRowI = blockIdx.x * 32;
    const int tid = threadIdx.x;
    const int localNRowIs = min(32, nRowIs-startTargetRowI);

    // cooperatively load 32 row indices
    if (tid < localNRowIs){
        sourceRowIndices[tid] = int(indices[startTargetRowI + tid]);
        if (sourceRowIndices[tid]<0)
            sourceRowIndices[tid] += nSourceRows;
        if (sourceRowIndices[tid]<0 || sourceRowIndices[tid]>=nSourceRows)
            sourceRowIndices[tid] = -1;
    }
    __syncthreads();

    // copy 32 rows
    for (int i=0; i<localNRowIs; i++){
        const int targetRowI = startTargetRowI + i, sourceRowI = sourceRowIndices[i];
        for (int colI=tid; colI<nCols; colI+=32)
            target[targetRowI * nCols + colI] = sourceRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
    }
}

__global__ void kSetSelectedRows(double* target, double* source, double* indices, int nRowIs, int nCols, int nTargetRows){
    __shared__ int targetRowIndices[32];
    const int startSourceRowI = blockIdx.x * 32;
    const int tid = threadIdx.x;
    const int localNRowIs = min(32, nRowIs-startSourceRowI);

    // cooperatively load 32 row indices
    if (tid < localNRowIs){
        targetRowIndices[tid] = int(indices[startSourceRowI + tid]);
        if (targetRowIndices[tid]<0)
            targetRowIndices[tid] += nTargetRows;
        if (targetRowIndices[tid]<0 || targetRowIndices[tid]>=nTargetRows)
            targetRowIndices[tid] = -1;
    }
    __syncthreads();

    // copy 32 rows
    for (int i=0; i<localNRowIs; i++){
        const int sourceRowI = startSourceRowI + i, targetRowI = targetRowIndices[i];
        for (int colI=tid; colI<nCols; colI+=32)
            target[targetRowI * nCols + colI] = targetRowI==-1 ? (1.0/0.0 -1.0/0.0) : source[sourceRowI * nCols + colI];
    }
}


__global__ void kWhere(double* condition_mat, double* if_mat, double* else_mat, double* target, unsigned int len) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;
    
    for (unsigned int i = idx; i < len; i += numThreads) {
        target[i] = condition_mat[i] ? if_mat[i] : else_mat[i];
    }
}


__global__ void kCorrelate(double* source, double* kernel, double* dest, int width, int height, int kwidth, int kheight) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numThreads = blockDim.x * gridDim.x;

    for (unsigned int i = idx; i < width * height; i += numThreads) {
        double sum = 0;
        for (int w = -kwidth/2; w <= kwidth/2; w++) {
            for (int h = -kheight/2; h <= (kheight)/2; h++) {
                const int x = (i / height) + w;
                const int y = (i % height) + h;
                const int j = i + (w * height) + h;

                if (x >= 0 && x < width && y >= 0 && y < height)
                    sum += source[j] * kernel[(kwidth * kheight / 2) + w * kheight + h];
            }
        }
        dest[i] = sum;
    }
}
