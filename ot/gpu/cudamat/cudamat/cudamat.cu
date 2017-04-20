#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <cublas.h>
#include "cudamat_kernels.cuh"
#include "cudamat.cuh"

extern "C" {

/* ------------------------------ CUBLAS init/shutdown ------------------------------ */

inline bool check_cublas_error() {
    cublasStatus status = cublasGetError();

    return status != CUBLAS_STATUS_SUCCESS;
}

inline bool checkCUDAError() {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        printf("%s\n", cudaGetErrorString( err));
    return cudaSuccess != err;
}

EXPORT const char* get_last_cuda_error() {
    cudaError_t err = cudaGetLastError();

    return cudaGetErrorString( err);
}

EXPORT const char* get_last_clib_error() {
    return strerror(errno);
}

EXPORT int cublas_init() {
    cublasInit();
    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

EXPORT int cublas_shutdown() {
    cublasShutdown();
    cudaThreadExit();

    return 0;
}


EXPORT int cuda_set_device(int deviceId) {
    cudaSetDevice(deviceId);
    
    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

EXPORT int init_random(rnd_struct* rnd_state, int seed, char* cudamatpath) {
    unsigned int * host_mults;
    host_mults = (unsigned int*)malloc(NUM_RND_STREAMS * sizeof(unsigned int));
    FILE * pFile;

    pFile = fopen (cudamatpath,"r");
    if (pFile == NULL) {
        return ERROR_FILE_OPEN;
    }

    for (int i = 0; i < NUM_RND_STREAMS; i++) {
        if (fscanf (pFile, "%u", &host_mults[i]) != 1) {
            return ERROR_FILE_SCAN;
        }
    }
    fclose (pFile);

    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned int), (void**)&rnd_state->dev_mults);
    cublasAlloc(NUM_RND_STREAMS, sizeof(unsigned long long), (void**)&rnd_state->dev_words);
    cublasSetVector(NUM_RND_STREAMS, sizeof(unsigned int), host_mults, 1, rnd_state->dev_mults, 1);
    //cudaMalloc((void **)&rnd_state->dev_mults, NUM_RND_STREAMS * sizeof(unsigned int));
    //cudaMalloc((void **)&rnd_state->dev_words, NUM_RND_STREAMS * sizeof(unsigned long long));
    //cudaMemcpy(rnd_state->dev_mults, host_mults, NUM_RND_STREAMS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    kSeedRandom<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, seed);
 
    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

/* ------------------------------ Utility routines ------------------------------ */

EXPORT int get_leading_dimension(cudamat* mat) {
    return mat->is_trans ? mat->size[1] : mat->size[0];
}

EXPORT int get_nonleading_dimension(cudamat* mat) {
    return mat->is_trans ? mat->size[0] : mat->size[1];
}

EXPORT void set_transpose(cudamat* mat, int is_trans) {
    mat->is_trans = is_trans;
}

inline char get_transpose_char(cudamat* mat) {
    return mat->is_trans ? 't' : 'n';
}

EXPORT void cuda_sync_threads() {
    cudaThreadSynchronize();
}

/* ------------------------------ Allocating/moving data ------------------------------ */

EXPORT int allocate_device_memory(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];

    cublasStatus stat;

    stat = cublasAlloc(len, sizeof(mat->data_device[0]), (void**)&mat->data_device);

    if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error()) {
        checkCUDAError();
        return CUBLAS_ERROR;
    }

    mat->on_device = 1;
    return 0;
}

EXPORT int copy_to_host(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];

    if (mat->on_device) {
            cublasGetVector(len, sizeof(mat->data_host[0]), mat->data_device, 1, mat->data_host, 1);

        if (check_cublas_error())
            return CUBLAS_ERROR;
    } else
       return ERROR_NOT_ON_DEVICE;
 
    return 0;
}

EXPORT int copy_to_device(cudamat* mat) {
    int len = mat->size[0]*mat->size[1];
    int err_code = 0;

    //if (!mat->owns_data)
    //    return VIEW_ERROR;

    if (!mat->on_device) {
        err_code = allocate_device_memory(mat);
        if (err_code)
            return err_code;
    }

    cublasSetVector(len, sizeof(mat->data_host[0]), mat->data_host, 1, mat->data_device, 1);
    
    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

EXPORT int copy_on_device(cudamat* mat1, cudamat* mat2) {
    int len = mat1->size[0]*mat1->size[1];

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cublasDcopy(len, mat1->data_device, 1, mat2->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;
    else
        return 0;
}

EXPORT int get_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end) {
    int height = source->size[0];
    int width = source->size[1];

    if ((end - start) != target->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    dim3 kernelBlockGrid((int)ceil((end - start)/32.), (int)ceil(width/32.), 1);
    dim3 kernelBlockDim(32, 1, 1);

    kGetRowSlice<<<kernelBlockGrid,kernelBlockDim>>>(source->data_device, target->data_device, start, end, width, height);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

EXPORT int set_row_slice(cudamat* source, cudamat* target, unsigned int start, unsigned int end) {
    int height = target->size[0];
    int width = target->size[1];

    if ((end - start) != source->size[0] || source->size[1] != target->size[1] || start >= end || end > height)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    dim3 kernelBlockGrid((int)ceil((end - start)/32.), (int)ceil(width/32.), 1);
    dim3 kernelBlockDim(32, 1, 1);

    kSetRowSlice<<<kernelBlockGrid,kernelBlockDim>>>(source->data_device, target->data_device, start, end, width, height);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

EXPORT int copy_transpose(cudamat* source, cudamat* target) {
    unsigned int height = source->size[0];
    unsigned int width = source->size[1];

    if (source->size[0] != target->size[1] || source->size[1] != target->size[0])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    // setup execution parameters
    unsigned int grid_x = height / COPY_BLOCK_SIZE;
    if (height % COPY_BLOCK_SIZE)
        grid_x++;

    unsigned int grid_y = width / COPY_BLOCK_SIZE;
    if (width % COPY_BLOCK_SIZE)
        grid_y++;

    dim3 grid(grid_x, grid_y, 1);
    dim3 threads(COPY_BLOCK_SIZE, COPY_BLOCK_SIZE, 1);

    kTranspose<<< grid, threads >>>(target->data_device, source->data_device, height, width);

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

EXPORT int free_device_memory(cudamat* mat) {
    if (mat->owns_data && mat->on_device) {
        cublasStatus stat;

        stat = cublasFree(mat->data_device);
        mat->on_device = 0;

        if (stat != CUBLAS_STATUS_SUCCESS || check_cublas_error())
            return CUBLAS_ERROR;
    }

    return 0;
}

EXPORT int reshape(cudamat* mat, unsigned int m, unsigned int n) {
    if (mat->size[0] * mat->size[1] != m * n)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    mat->size[0] = m;
    mat->size[1] = n;

    return 0;
}

EXPORT int get_slice(cudamat* source, cudamat* target, unsigned int first_col, unsigned int last_col) {
    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (last_col > source->size[1] || (first_col >= last_col))
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = source->data_device + first_col * num_rows;
    target->on_device = 1;
    target->on_host = 0;
    target->size[0] = source->size[0];
    target->size[1] = last_col - first_col;
    target->is_trans = 0;
    target->owns_data = 0;

    return 0;
}

EXPORT int get_vector_slice(cudamat* source, cudamat* target, unsigned int first_ind, unsigned int last_ind) {
    // source must be a vector
    if (source->size[0] > 1 && source->size[1] > 1)
        return ERROR_GENERIC;

    if (source->is_trans)
        return ERROR_TRANSPOSED;

    if (!source->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (first_ind >= last_ind)
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    int num_rows = source->size[0];

    target->data_host = 0;
    target->data_device = source->data_device + first_ind * num_rows;
    target->on_device = 1;
    target->on_host = 0;
    target->is_trans = 0;
    target->owns_data = 0;

    if (source->size[0] > 1) {
        if (last_ind > source->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = last_ind - first_ind;
        target->size[1] = 1;
    } else {
        if (last_ind > source->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        target->size[0] = 1;
        target->size[1] = last_ind - first_ind;
    }

    return 0;
}

/* ------------------------------ Initialization routines ------------------------------ */

EXPORT void init_from_array(cudamat* mat, double* data, int m, int n) {
    mat->data_host = data;
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 1;
    mat->is_trans = 0;
    mat->owns_data = 1;
}

EXPORT int init_empty(cudamat* mat, int m, int n) {
    mat->size[0] = m;
    mat->size[1] = n;
    mat->on_device = 0;
    mat->on_host = 0;
    mat->is_trans = 0;
    mat->owns_data = 1;

    return allocate_device_memory(mat);
}

/* ------------------------------ Random number generation ------------------------------ */
EXPORT int fill_with_rand(rnd_struct* rnd_state, cudamat* mat) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kRandomUniform<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

EXPORT int fill_with_randn(rnd_struct* rnd_state, cudamat* mat) {
    int len = mat->size[0] * mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kRandomGaussian<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(rnd_state->dev_mults, rnd_state->dev_words, mat->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}
/* ------------------------------ Algebraic operations ------------------------------ */

EXPORT int add_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddColVector<<<NUM_VECTOR_OP_BLOCKS(w*h),NUM_VECTOR_OP_THREADS_PER_BLOCK(w*h)>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError()) {
        return CUDA_ERROR;
    }

    return 0;
}

EXPORT int add_col_mult(cudamat* mat, cudamat* vec, cudamat* target, double mult) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddColMult<<<NUM_VECTOR_OP_BLOCKS(w*h),NUM_VECTOR_OP_THREADS_PER_BLOCK(w*h)>>>(mat->data_device, vec->data_device, target->data_device, mult, w, h);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int add_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddRowVector<<<NUM_VECTOR_OP_BLOCKS(w*h),NUM_VECTOR_OP_THREADS_PER_BLOCK(w*h)>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int mult_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultByColVector<<<NUM_VECTOR_OP_BLOCKS(w*h),NUM_VECTOR_OP_THREADS_PER_BLOCK(w*h)>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int mult_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMultByRowVector<<<NUM_VECTOR_OP_BLOCKS(w*h),NUM_VECTOR_OP_THREADS_PER_BLOCK(w*h)>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int divide_by_col_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[0] != vec->size[0] || vec->size[1] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivByColVector<<<NUM_VECTOR_OP_BLOCKS(w*h),NUM_VECTOR_OP_THREADS_PER_BLOCK(w*h)>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int divide_by_row_vec(cudamat* mat, cudamat* vec, cudamat* target) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !vec->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (mat->size[1] != vec->size[1] || vec->size[0] != 1 ||
        mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivByRowVector<<<NUM_VECTOR_OP_BLOCKS(w*h),NUM_VECTOR_OP_THREADS_PER_BLOCK(w*h)>>>(mat->data_device, vec->data_device, target->data_device, w, h);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int less_than(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLessThan<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int less_than_scalar(cudamat* mat, double val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLessThanScalar<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, val, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int greater_than(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kGreaterThan<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int greater_than_scalar(cudamat* mat, double val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kGreaterThanScalar<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, val, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int equals(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kEquals<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int equals_scalar(cudamat* mat, double val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kEqualsScalar<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, val, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int minimum(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMinimum<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int minimum_scalar(cudamat* mat, double val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMinimumScalar<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int maximum(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMaximum<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int maximum_scalar(cudamat* mat, double val, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMaximumScalar<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, val, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int min_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kMinColumnwise<<<w,32>>>(mat->data_device, target->data_device, w, h);
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kMinRowwise<<<h,32>>>(mat->data_device, target->data_device, w, h);
    }

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int max_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kMaxColumnwise<<<w,32>>>(mat->data_device, target->data_device, w, h);
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kMaxRowwise<<<h,32>>>(mat->data_device, target->data_device, w, h);
    }

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int argmin_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kArgMinColumnwise<<<w,32>>>(mat->data_device, target->data_device, w, h);
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kArgMinRowwise<<<h,32>>>(mat->data_device, target->data_device, w, h);
    }

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int argmax_by_axis(cudamat* mat, cudamat* target, int axis) {
    unsigned int h = mat->size[0],
                 w = mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans)
        return ERROR_TRANSPOSED;

    if (axis == 0) {
        if (target->size[0] != 1 || target->size[1] != mat->size[1])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kArgMaxColumnwise<<<w,32>>>(mat->data_device, target->data_device, w, h);
    } else {
        if (target->size[1] != 1 || target->size[0] != mat->size[0])
            return ERROR_INCOMPATIBLE_DIMENSIONS;

        kArgMaxRowwise<<<h,32>>>(mat->data_device, target->data_device, w, h);
    }

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int sign(cudamat* mat, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->is_trans != target->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSign<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_sigmoid(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplySigmoid<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_tanh(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplyTanh<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_soft_threshold(cudamat* mat, double alpha, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplySoftThreshold<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, alpha, target->data_device, len);

    cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_abs(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplyAbs<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_log_1_plus_exp(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kApplyLog1PlusExp<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_log(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLog<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_exp(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kExp<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_gamma(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kGamma<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_lgamma(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kLogGamma<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_sqrt(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSqrt<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_pow(cudamat* mat, double pow, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kPow<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, pow, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int apply_pow_matrix(cudamat* mat, cudamat* pow, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat->size[0] != pow->size[0] || mat->size[1] != pow->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kPowMatrix<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, pow->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int reciprocal(cudamat* mat, cudamat* target) {
    unsigned int len = mat->size[0] * mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kReciprocal<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int dot(cudamat* mat1, cudamat* mat2, cudamat* target, double beta, double alpha) {
    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (get_leading_dimension(mat1) != get_leading_dimension(target) ||
        get_nonleading_dimension(mat2) != get_nonleading_dimension(target) ||
        get_nonleading_dimension(mat1) != get_leading_dimension(mat2)) {
        return ERROR_INCOMPATIBLE_DIMENSIONS;
    }
    int m = get_leading_dimension(mat1),
        k = get_leading_dimension(mat2),
        n = get_nonleading_dimension(mat2);

    // gemv if second matrix is a (column) vector
    if (n == 1) {
        cublasDgemv(get_transpose_char(mat1), mat1->size[0], mat1->size[1],
                    alpha, mat1->data_device, mat1->size[0],
                    mat2->data_device, 1,
                    beta, target->data_device, 1);
    }
    // gemv if first matrix is a (row) vector
    else if (m == 1) {
        cublasDgemv(mat2->is_trans ? 'n' : 't', mat2->size[0], mat2->size[1],
                    alpha, mat2->data_device, mat2->size[0],
                    mat1->data_device, 1,
                    beta, target->data_device, 1);
    }
    // gemm otherwise
    else {
        cublasDgemm(get_transpose_char(mat1), get_transpose_char(mat2),
                    m, n, k,
                    alpha, mat1->data_device, mat1->size[0],
                    mat2->data_device, mat2->size[0],
                    beta, target->data_device, target->size[0]);
    }

    if (check_cublas_error())
        return CUBLAS_ERROR;

    if (SYNC_THREADS) 
        cudaThreadSynchronize();

    return 0;
}

EXPORT double vdot(cudamat* mat1, cudamat* mat2, int* err_code) {
    int len = mat1->size[0]*mat1->size[1];
    double res;

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans) {
        *err_code = ERROR_TRANSPOSEDNESS;
        return 0;
    }

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1]) { 
        *err_code = ERROR_INCOMPATIBLE_DIMENSIONS;
        return 0;
    }

    res = cublasDdot(len, mat1->data_device, 1, mat2->data_device, 1);

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}

/* Perform the operation mat1 = mat1 + alpha * mat2. mat1 and mat2 must
   have the same transposedness. */
EXPORT int add_mult(cudamat* mat1, cudamat* mat2, double alpha) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    cublasDaxpy(len, alpha, mat2->data_device, 1, mat1->data_device, 1);

    if (check_cublas_error())
        return CUBLAS_ERROR;

    return 0;
}

EXPORT int add_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat1 == target) {
        cublasDaxpy(len, 1, mat2->data_device, 1, mat1->data_device, 1);
 
        if (check_cublas_error())
            return CUBLAS_ERROR;

    } else {
        kAdd<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat1->data_device, mat2->data_device, target->data_device, len);
 
        if (SYNC_THREADS)
            cudaThreadSynchronize();

        if (checkCUDAError())
            return CUDA_ERROR;
    }
 
     return 0;
}

EXPORT int subtract_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kSubtract<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int divide_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivide<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

/* Elementwise multiplication of 2 matrices */
EXPORT int mult_elementwise(cudamat* mat1, cudamat* mat2, cudamat* target) {
    int len = mat1->size[0]*mat1->size[1];

    if (!mat1->on_device || !mat2->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat1->is_trans != mat2->is_trans)
        return ERROR_TRANSPOSEDNESS;

    if (mat1->size[0] != mat2->size[0] || mat1->size[1] != mat2->size[1] ||
        mat1->size[0] != target->size[0] || mat1->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kMult<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat1->data_device, mat2->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int assign_scalar(cudamat* mat, double alpha) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device)
        return ERROR_NOT_ON_DEVICE;

    kAssignScalar<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, alpha, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int mult_by_scalar(cudamat* mat, double alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (mat == target) {
        cublasDscal(len, alpha, mat->data_device, 1);
 
        if (check_cublas_error())
            return CUBLAS_ERROR;

    } else {
        kMultScalar<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, alpha, target->data_device, len);

        if (SYNC_THREADS) 
            cudaThreadSynchronize();

        if (checkCUDAError())
            return CUDA_ERROR;
    }
 
    return 0;
}

EXPORT int divide_by_scalar(cudamat* mat, double alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kDivideScalar<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, alpha, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int add_scalar(cudamat* mat, double alpha, cudamat* target) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (mat->size[0] != target->size[0] || mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kAddScalar<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(mat->data_device, alpha, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT double euclid_norm(cudamat* mat, int* err_code) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device) {
        *err_code = ERROR_NOT_ON_DEVICE;    
        return -1.;
    }

    double res = cublasDnrm2(len, mat->data_device, 1);

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}

EXPORT double manhattan_norm(cudamat* mat, int* err_code) {
    int len = mat->size[0]*mat->size[1];

    if (!mat->on_device) {
        *err_code = ERROR_NOT_ON_DEVICE;    
        return -1.;
    }

    double res = cublasDasum(len, mat->data_device, 1);

    if (check_cublas_error()) {
        *err_code = CUBLAS_ERROR;
        return -1.;
    } else {
        *err_code = 0;
        return res;
    }
}

EXPORT int selectRows(cudamat* source, cudamat* target, cudamat* indices){
    const int nRetRows = indices->size[1];

    if (nRetRows==0) return 0;

    dim3 gridDim((nRetRows+31)/32);
    dim3 blockDim(32);

    kSelectRows<<<gridDim, blockDim>>>(source->data_device, target->data_device, indices->data_device, nRetRows, source->size[0], source->size[1]);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

EXPORT int setSelectedRows(cudamat* target, cudamat* source, cudamat* indices){
    const int nSetRows = indices->size[1];

    if (nSetRows==0)
        return 0;

    dim3 gridDim((nSetRows+31)/32);
    dim3 blockDim(32);

    kSetSelectedRows<<<gridDim, blockDim>>>(target->data_device, source->data_device, indices->data_device, nSetRows, target->size[0], target->size[1]);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;
    else
        return 0;
}

EXPORT int where(cudamat* condition_mat, cudamat* if_mat, cudamat* else_mat, cudamat* target) {
    unsigned int len = condition_mat->size[0] * condition_mat->size[1];

    if (!condition_mat->on_device || !target->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (condition_mat->size[0] != target->size[0] || condition_mat->size[1] != target->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (condition_mat->size[0] != if_mat->size[0] || condition_mat->size[1] != if_mat->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;
        
    if (condition_mat->size[0] != else_mat->size[0] || condition_mat->size[1] != else_mat->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kWhere<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(condition_mat->data_device,
        if_mat->data_device, else_mat->data_device, target->data_device, len);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

    return 0;
}

EXPORT int correlate(cudamat* source, cudamat* kernel, cudamat* dest) {
    int len = source->size[0] * source->size[1];

    if (!source->on_device || !kernel->on_device || !dest->on_device)
        return ERROR_NOT_ON_DEVICE;

    if (source->size[0] != dest->size[0] || source->size[1] != dest->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    if (kernel->size[0] % 2 == 0 || kernel->size[1] % 2 == 0 ||
        kernel->size[0] > source->size[0] || kernel->size[1] > source->size[1])
        return ERROR_INCOMPATIBLE_DIMENSIONS;

    kCorrelate<<<NUM_VECTOR_OP_BLOCKS(len),NUM_VECTOR_OP_THREADS_PER_BLOCK(len)>>>(source->data_device, 
        kernel->data_device, dest->data_device, source->size[1], source->size[0], 
        kernel->size[1], kernel->size[0]);

    if (SYNC_THREADS)
        cudaThreadSynchronize();

    if (checkCUDAError())
        return CUDA_ERROR;

     return 0;
}

}
