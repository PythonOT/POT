import os
import platform
import warnings
import sysconfig
import sys

import ctypes as ct
import numpy as np

def load_library(basename):
    if platform.system() == 'Windows':
       ext = '.dll'
    else:
       ext = sysconfig.get_config_var('SO')
    return ct.cdll.LoadLibrary(os.path.join(
        os.path.dirname(__file__) or os.path.curdir,
        basename + ext))

_cudamat = load_library('libcudamat')

_cudamat.get_last_cuda_error.restype = ct.c_char_p
_cudamat.get_last_clib_error.restype = ct.c_char_p
_cudamat.cublas_init.restype = ct.c_int
_cudamat.cublas_shutdown.restype = ct.c_int
_cudamat.cuda_set_device.restype = ct.c_int
_cudamat.init_random.restype = ct.c_int

_cudamat.init_empty.restype = ct.c_int
_cudamat.reshape.restype = ct.c_int
_cudamat.copy_to_host.restype = ct.c_int
_cudamat.allocate_device_memory = ct.c_int
_cudamat.copy_to_device.restype = ct.c_int
_cudamat.copy_on_device.restype = ct.c_int
_cudamat.free_device_memory.restype = ct.c_int

_cudamat.get_slice.restype = ct.c_int
_cudamat.get_row_slice.restype = ct.c_int
_cudamat.set_row_slice.restype = ct.c_int
_cudamat.copy_transpose.restype = ct.c_int
_cudamat.get_vector_slice.restype = ct.c_int
_cudamat.fill_with_rand.restype = ct.c_int
_cudamat.fill_with_randn.restype = ct.c_int

_cudamat.add_col_vec.restype = ct.c_int
_cudamat.add_col_mult.restype = ct.c_int
_cudamat.add_row_vec.restype = ct.c_int
_cudamat.mult_by_col_vec.restype = ct.c_int
_cudamat.mult_by_row_vec.restype = ct.c_int
_cudamat.divide_by_col_vec.restype = ct.c_int
_cudamat.divide_by_row_vec.restype = ct.c_int

_cudamat.less_than.restype = ct.c_int
_cudamat.less_than_scalar.restype = ct.c_int
_cudamat.greater_than.restype = ct.c_int
_cudamat.greater_than_scalar.restype = ct.c_int
_cudamat.equals.restype = ct.c_int
_cudamat.equals_scalar.restype = ct.c_int
_cudamat.minimum.restype = ct.c_int
_cudamat.minimum_scalar.restype = ct.c_int
_cudamat.maximum.restype = ct.c_int
_cudamat.maximum_scalar.restype = ct.c_int
_cudamat.min_by_axis.restype = ct.c_int
_cudamat.max_by_axis.restype = ct.c_int
_cudamat.argmin_by_axis.restype = ct.c_int
_cudamat.argmax_by_axis.restype = ct.c_int
_cudamat.sign.restype = ct.c_int
_cudamat.apply_sigmoid.restype = ct.c_int
_cudamat.apply_tanh.restype = ct.c_int
_cudamat.apply_soft_threshold.restype = ct.c_int
_cudamat.apply_abs.restype = ct.c_int
_cudamat.apply_log_1_plus_exp.restype = ct.c_int
_cudamat.apply_log.restype = ct.c_int
_cudamat.apply_exp.restype = ct.c_int
_cudamat.apply_gamma.restype = ct.c_int
_cudamat.apply_lgamma.restype = ct.c_int
_cudamat.apply_sqrt.restype = ct.c_int
_cudamat.apply_pow.restype = ct.c_int
_cudamat.apply_pow_matrix.restype = ct.c_int
_cudamat.reciprocal.restype = ct.c_int

_cudamat.add_elementwise.restype = ct.c_int
_cudamat.subtract_elementwise.restype = ct.c_int
_cudamat.divide_elementwise.restype = ct.c_int
_cudamat.mult_elementwise.restype = ct.c_int
_cudamat.assign_scalar.restype = ct.c_int
_cudamat.mult_by_scalar.restype = ct.c_int
_cudamat.divide_by_scalar.restype = ct.c_int
_cudamat.add_scalar.restype = ct.c_int

_cudamat.euclid_norm.restype = ct.c_double
_cudamat.manhattan_norm.restype = ct.c_double
_cudamat.selectRows.restype = ct.c_int
_cudamat.setSelectedRows.restype = ct.c_int
_cudamat.vdot.restype = ct.c_double
_cudamat.dot.restype = ct.c_int

_cudamat.where.restype = ct.c_int

_cudamat.correlate.restype = ct.c_int


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def newFunc(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc


class CUDAMatException(Exception):
    pass


def get_last_cuda_error():
    errmsg = _cudamat.get_last_cuda_error()
    if sys.version_info >= (3,):
        return bytes(errmsg).decode()
    else:
        return str(errmsg)


def get_last_clib_error():
    errmsg = _cudamat.get_last_clib_error()
    if sys.version_info >= (3,):
        return bytes(errmsg).decode()
    else:
        return str(errmsg)


def generate_exception(err_code, **kwargs):
    """
    Return a CUDAMatException object based on the error code err_code.
    Additional arguments are error-specific and optional.
    """

    if err_code == -1:
        return CUDAMatException("Incompatible matrix dimensions.")
    elif err_code == -2:
        return CUDAMatException("CUBLAS error.")
    elif err_code == -3:
        return CUDAMatException("CUDA error: " + get_last_cuda_error())
    elif err_code == -4:
        return CUDAMatException("Operation not supported on views.")
    elif err_code == -5:
        return CUDAMatException("Operation not supported on "
                                "transposed matrices.")
    elif err_code == -6:
        return CUDAMatException("")
    elif err_code == -7:
        return CUDAMatException("Incompatible transposedness.")
    elif err_code == -8:
        return CUDAMatException("Matrix is not in device memory.")
    elif err_code == -9:
        return CUDAMatException("Operation not supported.")
    elif err_code == -10:
        filepath = kwargs.get("filepath","");
        if filepath:
            filepath = ": '%s'" % filepath
        return CUDAMatException("Cannot open file%s: %s" % (filepath,get_last_clib_error()))
    elif err_code == -11:
        filepath = kwargs.get("filepath","");
        if filepath:
            filepath = ": '%s'" % filepath
        return CUDAMatException("Cannot parse file%s." % filepath)
    else:
        return CUDAMatException("")


class cudamat(ct.Structure):
    _fields_ = [('data_host', ct.POINTER(ct.c_double)),
                ('data_device', ct.POINTER(ct.c_double)),
                ('on_device', ct.c_int),
                ('on_host', ct.c_int),
                ('size', ct.c_int * 2),
                ('is_trans', ct.c_int),
                ('owns_data', ct.c_int)]


class rnd_struct(ct.Structure):
    _fields_ = [('dev_rnd_mults', ct.POINTER(ct.c_uint)),
                ('dev_rnd_words', ct.POINTER(ct.c_longlong))]


class TransposedCUDAMatrix(object):
    def __init__(self, mat):
        self.mat = cudamat()
        ct.memmove(ct.pointer(self.mat), ct.pointer(mat), ct.sizeof(self.mat))
        self.mat.is_trans = 1
        self.p_mat = ct.pointer(self.mat)


class CUDAMatrix(object):
    """
    A CUDAMatrix object represents a matrix of single precision floating point
    numbers on a GPU.
    """

    def __init__(self, array, copy_to_device=True, copy_on_host=True):
        """
        Initializes a new matrix object in one of two ways. If array is a numpy
        ndarray, memory for a matrix with the same dimensions is allocated on
        the GPU. If the copy_to_device flag is set to True, the GPU matrix is
        initialized with the given ndarray. If the copy_on_host flag is set to
        True, a copy of the matrix will be created in host memory even if the
        matrix is of the correct type (float64, Fortran-contiguous order).
        If array is not an ndarray, it must be a cudamat structure (typically
        the user will never use this way of calling __init__).
        """

        if type(array) in [np.ndarray, np.memmap]:
            # Convert array to float64 in FORTRAN order
            array = reformat(array, copy=copy_on_host)

            # Initialize as a ndarray-tied matrix.
            self.mat = cudamat()
            self.size = self.mat.size
            self.p_mat = ct.pointer(self.mat)
            self.numpy_array = array

            _cudamat.init_from_array(
                self.p_mat,
                array.ctypes.data_as(ct.POINTER(ct.c_double)),
                ct.c_int(array.shape[0]),
                ct.c_int(array.shape[1]))

            if copy_to_device:
                err_code = _cudamat.copy_to_device(self.p_mat)
                if err_code:
                    raise generate_exception(err_code)

        else:
            # Initialize based on existing cudamat structure.
            mat = array
            self.mat = mat
            self.p_mat = ct.pointer(self.mat)

        self.T = TransposedCUDAMatrix(self.mat)

        # Keep a reference to free device memory in case of a crash.
        self.__free_device_memory = _cudamat.free_device_memory

    def __del__(self):
        try:
            if 'p_mat' in self.__dict__:
                err_code = self.__free_device_memory(self.p_mat)
                if err_code:
                    raise generate_exception(err_code)
        except AttributeError:
            pass

    @staticmethod
    def init_random(seed=0):
        """
        Initialize and seed the random number generator.
        """

        CUDAMatrix.rndInitialized = 1
        CUDAMatrix.rnd_state = rnd_struct()
        CUDAMatrix.rnd_state_p = ct.pointer(CUDAMatrix.rnd_state)

        cudamat_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                    'rnd_multipliers_32bit.txt')

        if sys.version_info >= (3,):
            cudamat_path = cudamat_path.encode(sys.getfilesystemencoding())

        err_code = _cudamat.init_random(CUDAMatrix.rnd_state_p,
                                        ct.c_int(seed),
                                        cudamat_path)
        if err_code:
            if sys.version_info >= (3,):
                cudamat_path = cudamat_path.decode(sys.getfilesystemencoding())
            raise generate_exception(err_code, filepath=cudamat_path)

    @property
    def shape(self):
        return (self.mat.size[0], self.mat.size[1])

    def reshape(self, shape):
        """
        Reshapes self to have the given shape. The number of elements cannot
        change as this only changes how the contents are interpreted.
        """

        m = ct.c_uint(shape[0])
        n = ct.c_uint(shape[1])

        # Reshape the default matrix
        err_code = _cudamat.reshape(self.p_mat, m, n)
        if err_code:
            raise generate_exception(err_code)
        # Reshape the transposed matrix
        err_code = _cudamat.reshape(self.T.p_mat, m, n)
        if err_code:
            raise generate_exception(err_code)
        # Reshape the CPU matrix
        if self.mat.on_host:
            self.numpy_array = np.reshape(self.numpy_array, shape, order='F')

        return self

    def asarray(self):
        """
        Copies the matrix to an ndarray on the CPU and returns it.
        """

        self.copy_to_host()

        return self.numpy_array

    def copy_to_device(self):
        """
        Copy the matrix to the GPU.
        """

        err_code = _cudamat.copy_to_device(self.p_mat)
        if err_code:
            raise generate_exception(err_code)

    def copy_to_host(self):
        """
        Copy the matrix to the CPU.
        """

        if not self.mat.on_host:
            # allocate host storage if necessary
            m = self.mat.size[0]
            n = self.mat.size[1]

            self.numpy_array = np.empty((m, n), dtype=np.float64, order='F')
            self.mat.data_host = \
                self.numpy_array.ctypes.data_as(ct.POINTER(ct.c_double))

            self.mat.on_host = 1

        err_code = _cudamat.copy_to_host(self.p_mat)
        if err_code:
            raise generate_exception(err_code)

    def copy(self, include_host=False):
        """
        Create a copy of the matrix on GPU. If include_host is True, also
        creates a copy of the matrix on CPU if there was any.
        """

        new_mat = empty(self.shape).assign(self)

        if include_host and self.mat.on_host:
            new_mat.numpy_array = self.numpy_array.copy()
            new_mat.mat.data_host = \
                new_mat.numpy_array.ctypes.data_as(ct.POINTER(ct.c_double))
            new_mat.mat.on_host = 1

        return new_mat

    def assign(self, val):
        """Assign val to self, where val can be a scalar or a CUDAMatrix
        with the same dimensions as self. """

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.copy_on_device(val.p_mat, self.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.assign_scalar(self.p_mat, ct.c_double(val))
        else:
            raise ValueError("Assigned value must be of type"
                             "CUDAMatrix, int, or float.")

        if err_code:
            raise generate_exception(err_code)

        return self

    def free_device_memory(self):
        """
        Free memory used up by the matrix on the GPU.
        """

        err_code = _cudamat.free_device_memory(self.p_mat)
        if err_code:
            raise generate_exception(err_code)

    def set_trans(self, is_trans):
        """
        Set the transposedness flag to is_trans.
        """

        _cudamat.set_transpose(self.p_mat, ct.c_int(1 * is_trans))

    def slice(self, first_col, last_col, include_host=False):
        """
        Creates a view into a consecutive range of columns of an existing
        matrix on GPU. If include_host is set to True, also creates a view
        into the CPU copy of the matrix (i.e., the numpy_array).
        """
        mat = cudamat()

        if self.mat.size[0] == 1 or self.mat.size[1] == 1:
            err_code = _cudamat.get_vector_slice(self.p_mat,
                                                 ct.pointer(mat),
                                                 ct.c_int(first_col),
                                                 ct.c_int(last_col))
        else:
            err_code = _cudamat.get_slice(self.p_mat,
                                          ct.pointer(mat),
                                          ct.c_int(first_col),
                                          ct.c_int(last_col))

        if err_code:
            raise generate_exception(err_code)

        new_mat = CUDAMatrix(mat)

        try:
            new_mat.sliceof = self.sliceof
        except:
            new_mat.sliceof = self

        # reproduce the slice on the host as well (if requested)
        if include_host and self.mat.on_host:
            new_mat.numpy_array = self.numpy_array[:, first_col:last_col]
            new_mat.mat.data_host = \
                new_mat.numpy_array.ctypes.data_as(ct.POINTER(ct.c_double))
            new_mat.mat.on_host = 1

        return new_mat

    def get_col_slice(self, first_col, last_col, target=None):
        """
        Get the columns with indices first_col through last_col. If a target
        is provided, columns are copied into the target. Otherwise, returns a
        view into the existing memory on GPU.
        """
        col_slice = self.slice(first_col, last_col)

        if target:
            target.assign(col_slice)
            return target
        else:
            return col_slice

    def set_col_slice(self, first_col, last_col, mat):
        """
        Assign the contents of mat to the columns with indices first_col
        through last_col.
        """
        self.slice(first_col, last_col).assign(mat)

        return self

    def get_row_slice(self, start, end, target=None):
        """
        Get the rows with indices start through end. If target is not provided
        memory for a new matrix will be allocated.
        """

        width = self.shape[1]

        if not target:
            target = empty((end-start, width))

        err_code = _cudamat.get_row_slice(self.p_mat,
                                          target.p_mat,
                                          ct.c_int(start),
                                          ct.c_int(end))
        if err_code:
            raise generate_exception(err_code)

        return target

    def set_row_slice(self, start, end, mat):
        """
        Assign the contents of mat to the rows with indices start through end.
        """

        err_code = _cudamat.set_row_slice(mat.p_mat, self.p_mat,
                                          ct.c_int(start), ct.c_int(end))
        if err_code:
            raise generate_exception(err_code)

        return self

    def transpose(self, target=None):
        """
        Return a transposed copy of the matrix.
        """
        if not target:
            target = empty((self.shape[1], self.shape[0]))

        err_code = _cudamat.copy_transpose(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def fill_with_rand(self):
        """
        Fill matrix on the GPU with random numbers drawn from the uniform
        distribution over the (0,1) interval.
        """

        err_code = _cudamat.fill_with_rand(CUDAMatrix.rnd_state_p, self.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return self

    def fill_with_randn(self):
        """
        Fill matrix on the GPU with random numbers drawn from
        the standard normal distribution.
        """

        err_code = _cudamat.fill_with_randn(CUDAMatrix.rnd_state_p, self.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return self

    def add_col_vec(self, vec, target=None):
        """
        Add vector vec to every column of the matrix. If a target is provided,
        it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.add_col_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def add_col_mult(self, vec, mult, target=None):
        """
        Add a multiple of vector vec to every column of the matrix. If a target
        is provided, it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.add_col_mult(self.p_mat, vec.p_mat,
                                         target.p_mat, ct.c_double(mult))
        if err_code:
            raise generate_exception(err_code)

        return target

    def add_row_vec(self, vec, target=None):
        """
        Add vector vec to every row of the matrix. If a target is provided,
        it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.add_row_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def mult_by_col(self, vec, target=None):
        """
        Multiply vector vec into every column of the matrix. If a target is
        provided, it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.mult_by_col_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def mult_by_row(self, vec, target=None):
        """
        Multiply vector vec into every row of the matrix. If a target is
        provided, it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.mult_by_row_vec(self.p_mat, vec.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def div_by_col(self, vec, target=None):
        """
        Divide every column of the matrix by vector vec. If a target is
        provided, it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.divide_by_col_vec(self.p_mat, vec.p_mat,
                                              target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def div_by_row(self, vec, target=None):
        """
        Divide every row of the matrix by vector vec. If a target is
        provided, it is used to store the result instead of self.
        """

        if not target:
            target = self

        err_code = _cudamat.divide_by_row_vec(self.p_mat, vec.p_mat,
                                              target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def sum(self, axis, target=None, mult=1.):
        """
        Sum the matrix along the given dimension, where 0 represents the leading
        dimension and 1 represents the non-leading dimension. If a target is
        not provided, a new vector is created for storing the result. The result
        is multiplied by the given factor mult (defaults to 1).
        """

        return sum(self, axis, target, mult)

    def mean(self, axis, target=None):
        """
        Compute the mean of the matrix along the given dimension, where 0
        represents the leading dimension and 1 represents the non-leading
        dimension. If a target is not provided, a new vector is created for
        storing the result.
        """

        return mean(self, axis, target)

    def add_sums(self, mat, axis, mult=1., beta=1.):
        """
        Add a multiple of the sums of the matrix mat along the given dimension
        to self. Self is scaled by beta before adding anything.
        """

        m = _cudamat.get_leading_dimension(mat.p_mat)
        n = _cudamat.get_nonleading_dimension(mat.p_mat)

        if axis == 0:
            # sum along leading dimension
            check_ones_matrix(m)
            left = CUDAMatrix.ones.slice(0, m)
            left.set_trans(True)
            right = mat

        elif axis == 1:
            # sum along non-leading dimension
            left = mat
            check_ones_matrix(n)
            right = CUDAMatrix.ones.slice(0, n)

        err_code = _cudamat.dot(left.p_mat, right.p_mat, self.p_mat,
                                ct.c_double(beta), ct.c_double(mult))
        if err_code:
            raise generate_exception(err_code)

        return self

    def less_than(self, val, target=None):
        """
        Perform the operation target = 1. * (self < val),
        where val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (int, float)):
            err_code = _cudamat.less_than_scalar(self.p_mat, ct.c_double(val),
                                                 target.p_mat)
        else:
            err_code = _cudamat.less_than(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def greater_than(self, val, target=None):
        """
        Perform the operation target = 1. * (self > val),
        where val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (int, float)):
            err_code = _cudamat.greater_than_scalar(self.p_mat,
                                                    ct.c_double(val),
                                                    target.p_mat)
        else:
            err_code = _cudamat.greater_than(self.p_mat, val.p_mat,
                                             target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def equals(self, val, target=None):
        """
        Perform the operation target = 1. * (self == val),
        where val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (int, float)):
            err_code = _cudamat.equals_scalar(self.p_mat, ct.c_double(val),
                                              target.p_mat)
        else:
            err_code = _cudamat.equals(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def minimum(self, val, target=None):
        """
        Perform the element-wise operation target = min(self, val), where
        val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (int, float)):
            err_code = _cudamat.minimum_scalar(self.p_mat, ct.c_double(val),
                                               target.p_mat)
        else:
            err_code = _cudamat.minimum(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def maximum(self, val, target=None):
        """
        Perform the element-wise operation target = max(self, val), where
        val can be a matrix or a scalar.
        """

        if not target:
            target = self

        if isinstance(val, (int, float)):
            err_code = _cudamat.maximum_scalar(self.p_mat, ct.c_double(val),
                                               target.p_mat)
        else:
            err_code = _cudamat.maximum(self.p_mat, val.p_mat, target.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def min(self, axis, target=None):
        """
        Find the minimum value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. If a
        target is not prvided, a new vector is created for storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))

        elif axis == 1:
            if not target:
                target = empty((m, 1))

        err_code = _cudamat.min_by_axis(self.p_mat, target.p_mat,
                                        ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def max(self, axis, target=None):
        """
        Find the maximum value along the given dimension, where 0 represents the
        leading dimension and 1 represents the non-leading dimension. If a
        target is not prvided, a new vector is created for storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))

        elif axis == 1:
            if not target:
                target = empty((m, 1))

        err_code = _cudamat.max_by_axis(self.p_mat, target.p_mat,
                                        ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def argmin(self, axis, target=None):
        """
        Find the index of the minimum value along the given dimension, where 0
        represents the leading dimension and 1 represents the non-leading
        dimension. If a target is not provided, a new vector is created for
        storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))

        elif axis == 1:
            if not target:
                target = empty((m, 1))

        err_code = _cudamat.argmin_by_axis(self.p_mat, target.p_mat,
                                           ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def argmax(self, axis, target=None):
        """
        Find the index of the maximum value along the given dimension, where 0
        represents the leading dimension and 1 represents the non-leading
        dimension. If a target is not provided, a new vector is created for
        storing the result.
        """

        m, n = self.shape

        if axis == 0:
            if not target:
                target = empty((1, n))

        elif axis == 1:
            if not target:
                target = empty((m, 1))

        err_code = _cudamat.argmax_by_axis(self.p_mat, target.p_mat,
                                           ct.c_int(axis))
        if err_code:
            raise generate_exception(err_code)

        return target

    def sign(self, target=None):
        """
        Find the sign of each element of the matrix.
        """

        if not target:
            target = empty((self.mat.size[0], self.mat.size[1]))

        err_code = _cudamat.sign(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def apply_sigmoid(self, target=None):
        """
        Apply the logistic sigmoid to each element of the matrix.
        """

        return sigmoid(self, target)

    def apply_tanh(self, target=None):
        """
        Apply the tanh to each element of the matrix.
        """

        return tanh(self, target)

    def apply_soft_threshold(self, alpha, target=None):
        """
        Apply the soft threshold function to each element of the matrix:

        x = sign(x) * max(0, abs(x) - alpha)
        """

        return soft_threshold(self, alpha, target)

    def reciprocal(self, target=None):
        """
        Find the reciprocal of each element of the matrix.
        """

        if not target:
            target = self

        err_code = _cudamat.reciprocal(self.p_mat, target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def dot(self, mat2, target=None):
        """
        Multiply the matrix by mat2 from the right.
        """

        return dot(self, mat2, target)

    def add_dot(self, m1, m2, mult=1., beta=1.):
        """
        Add the dot product of m1 and m2 to the matrix, scaled by mult.
        Self is scaled by beta before adding anything.
        """

        err_code = _cudamat.dot(m1.p_mat, m2.p_mat, self.p_mat,
                                ct.c_double(beta), ct.c_double(mult))
        if err_code:
            raise generate_exception(err_code)

        return self

    def subtract_dot(self, m1, m2, mult=1., beta=1.):
        """
        Subtract the dot product of m1 and m2 from the matrix, scaled by mult.
        Self is scaled by beta before subtracting anything.
        """

        return self.add_dot(m1, m2, mult=-1. * mult, beta=beta)

    def add_mult(self, mat2, alpha=1.):
        """
        Add multiple of mat2 to the matrix.
        """

        err_code = _cudamat.add_mult(self.p_mat, mat2.p_mat, ct.c_double(alpha))
        if err_code:
            raise generate_exception(err_code)

        return self

    def subtract_mult(self, mat2, alpha=1.):
        """
        Subtract a multiple of mat2 from the matrix.
        """

        err_code = _cudamat.add_mult(self.p_mat, mat2.p_mat,
                                     ct.c_double(-1. * alpha))
        if err_code:
            raise generate_exception(err_code)

        return self

    def add(self, val, target=None):
        """Add val to self, where val can be a scalar or a CUDAMatrix with the
        same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.add_elementwise(self.p_mat, val.p_mat,
                                                target.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.add_scalar(self.p_mat, ct.c_double(val),
                                           target.p_mat)
        else:
            raise ValueError("Value must be of type CUDAMatrix, int, or float.")

        if err_code:
            raise generate_exception(err_code)

        return target

    def subtract(self, val, target=None):
        """Subtract val from self, where val can be a scalar or a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.subtract_elementwise(self.p_mat, val.p_mat,
                                                     target.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.add_scalar(self.p_mat, ct.c_double(-1*val),
                                           target.p_mat)
        else:
            raise ValueError("Value must be of type CUDAMatrix, int, or float.")

        if err_code:
            raise generate_exception(err_code)

        return target

    def divide(self, val, target=None):
        """Divide self by val, where val can be a scalar or
        a CUDAMatrix with the same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.divide_elementwise(self.p_mat, val.p_mat,
                                                   target.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.divide_by_scalar(self.p_mat, ct.c_double(val),
                                                 target.p_mat)
        else:
            raise ValueError("Value must be of type CUDAMatrix, int, or float.")

        if err_code:
            raise generate_exception(err_code)

        return target

    def mult(self, val, target=None):
        """Multiply self by val, where val can be a scalar or a CUDAMatrix with
        the same dimensions as self. """

        if not target:
            target = self

        if isinstance(val, CUDAMatrix):
            err_code = _cudamat.mult_elementwise(self.p_mat, val.p_mat,
                                                 target.p_mat)
        elif isinstance(val, (int, float)):
            err_code = _cudamat.mult_by_scalar(self.p_mat, ct.c_double(val),
                                               target.p_mat)
        else:
            raise ValueError("Value must be of type CUDAMatrix, int, or float.")

        if err_code:
            raise generate_exception(err_code)

        return target

    @deprecated
    def assign_scalar(self, alpha):
        """
        Assign scalar alpha to every element of the matrix.
        """

        err_code = _cudamat.assign_scalar(self.p_mat, ct.c_double(alpha))
        if err_code:
            raise generate_exception(err_code)

        return self

    @deprecated
    def mult_by_scalar(self, alpha, target=None):
        """
        Multiply the matrix by a scalar.
        """

        if not target:
            target = self

        err_code = _cudamat.mult_by_scalar(self.p_mat, ct.c_double(alpha),
                                           target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    @deprecated
    def div_by_scalar(self, alpha, target=None):
        """
        Divide the matrix by a scalar.
        """

        if not target:
            target = self

        err_code = _cudamat.divide_by_scalar(self.p_mat, ct.c_double(alpha),
                                             target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    @deprecated
    def add_scalar(self, alpha, target=None):
        """
        Increment the matrix by a scalar.
        """

        if not target:
            target = self

        err_code = _cudamat.add_scalar(self.p_mat, ct.c_double(alpha),
                                       target.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return target

    def euclid_norm(self):
        """
        Returns the L2 norm of the matrix flattened to a vector.
        """
        err_code = ct.c_int(0)
        res = _cudamat.euclid_norm(self.p_mat, ct.byref(err_code))

        if err_code:
            raise generate_exception(err_code.value)

        return res

    def manhattan_norm(self):
        """
        Returns the L1 norm of the matrix flattened to a vector.
        """
        err_code = ct.c_int(0)
        res = _cudamat.manhattan_norm(self.p_mat, ct.byref(err_code))

        if err_code:
            raise generate_exception(err_code.value)

        return res

    def allfinite(self):
        """
        Checks if all entries in this matrix are finite, i.e., there is no
        NaN and no positive or negative infinity.
        """
        # Caveat: For a very large matrix of very large finite numbers, the
        # manhattan norm may overflow and allfinite() may return False.
        return np.isfinite(self.manhattan_norm())

    def select_columns(self, indices, target):
        """
        Copies some columns of self into target.
        <indices> must be a row vector. Its elements are float64's representing
        integers, e.g. "34.0" means the integer "34".
        After this call, for all r,c, target[r,c]=self[r,indices[c]].
        This returns target.
        Negative indices are interpreted in the usual Python way: all
        elements of <indices> had better be in the range
        [-self.shape[1], self.shape[1]-1].
        This does bounds checking, but out of bounds indices do not raise an
        exception (because the programmer was lazy). Instead, they result
        in NaN values in <target>.
        """

        err_code = _cudamat.selectRows(self.p_mat, target.p_mat, indices.p_mat)

        if err_code:
            raise generate_exception(err_code)

        return target

    def set_selected_columns(self, indices, source):
        """
        copies all columns of source into some columns of self.
        <indices> must be a row vector. Its elements are float64's representing
        integers, e.g. "34.0" means the integer "34". after this call, for all
        r,c, self[r,indices[c]]=source[r,c]. This returns self.
        Negative indices are interpreted in the usual Python way: all elements
        of <indices> had better be in the range
        [-self.shape[1], self.shape[1]-1].
        This does bounds checking, but out of bounds indices do not raise an
        exception (because the programmer was lazy). Instead, they result in NaN
        values in <self>.
        """

        err_code = _cudamat.setSelectedRows(self.p_mat, source.p_mat,
                                            indices.p_mat)
        if err_code:
            raise generate_exception(err_code)

        return self


def empty(shape):
    """
    Creates and returns a new CUDAMatrix with the given shape.
    """

    mat = cudamat()
    err_code = _cudamat.init_empty(ct.pointer(mat), ct.c_int(shape[0]),
                                   ct.c_int(shape[1]))

    if err_code:
        raise generate_exception(err_code)

    return CUDAMatrix(mat)


def check_ones_matrix(min_size):
    if min_size > CUDAMatrix.ones.shape[0]:
        raise CUDAMatException(
            'Not enough memory allocated for reduction. '
            '({} needed, {} actual), use cudamat.init() '
            'to allocate more'.format(min_size, CUDAMatrix.ones.shape[0]))


def sum(mat, axis, target=None, mult=1.):
    """
    Sum the matrix along the given dimension, where 0 represents the leading
    dimension and 1 represents the non-leading dimension. If a target is
    not provided, a new vector is created for storing the result. The result
    is multiplied by the given factor mult (defaults to 1).
    """

    m = _cudamat.get_leading_dimension(mat.p_mat)
    n = _cudamat.get_nonleading_dimension(mat.p_mat)

    if axis == 0:
        # sum along leading dimension
        check_ones_matrix(m)
        left = CUDAMatrix.ones.slice(0, m)
        left.set_trans(True)
        right = mat

        if not target:
            target = empty((1, n))

    elif axis == 1:
        # sum along non-leading dimension
        left = mat
        check_ones_matrix(n)
        right = CUDAMatrix.ones.slice(0, n)

        if not target:
            target = empty((m, 1))

    err_code = _cudamat.dot(left.p_mat, right.p_mat, target.p_mat,
                            ct.c_double(0.), ct.c_double(mult))
    if err_code:
        raise generate_exception(err_code)

    return target


def mean(mat, axis, target=None):
    """
    Compute the mean of the matrix along the given dimension, where 0 represents
    the leading dimension and 1 represents the non-leading dimension. If a
    target is not provided, a new vector is created for storing the result.
    """

    return sum(mat, axis, target=target, mult=1. / mat.shape[axis])


def dot(m1, m2, target=None, beta=0., alpha=1.):
    """
    Find the dot product between m1 and m2 and store in target:
    target = beta*target + alpha*(m1 m2)
    If no target is given, it will be created automatically, but not
    initialized -- so beta should be left at its default value zero.
    """

    if not target:
        m = _cudamat.get_leading_dimension(m1.p_mat)
        n = _cudamat.get_nonleading_dimension(m2.p_mat)

        target = empty((m, n))

    err_code = _cudamat.dot(m1.p_mat, m2.p_mat,
                            target.p_mat, ct.c_double(beta),
                            ct.c_double(alpha))
    if err_code:
        raise generate_exception(err_code)

    return target


def vdot(m1, m2):
    """
    Compute the vector dot product of matrices m1 and m2.
    """

    err_code = ct.c_int(0)
    res = _cudamat.vdot(m1.p_mat, m2.p_mat, ct.byref(err_code))

    if err_code:
        raise generate_exception(err_code.value)

    return res


def sigmoid(mat, target=None):
    """
    Apply the logistic sigmoid to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_sigmoid(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def tanh(mat, target=None):
    """
    Apply the tanh to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_tanh(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def soft_threshold(mat, alpha, target=None):
    """
    Apply the soft threshold function to each element of the matrix:

    mat = sign(mat) * max(0, abs(mat) - alpha)
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_soft_threshold(mat.p_mat, ct.c_double(alpha),
                                             target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def abs(mat, target=None):
    """
    Apply abs to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_abs(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def log_1_plus_exp(mat, target=None):
    """
    Apply log(1+exp(x)) to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_log_1_plus_exp(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def log(mat, target=None):
    """
    Find the natural logarithm of each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_log(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def exp(mat, target=None):
    """
    Apply the exponential function to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_exp(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def gamma(mat, target=None):
    """
    Apply the gamma function to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_gamma(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def lgamma(mat, target=None):
    """
    Apply the log gamma function to each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_lgamma(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def sqrt(mat, target=None):
    """
    Compute the square root of each element of the matrix mat.
    """

    if not target:
        target = mat

    err_code = _cudamat.apply_sqrt(mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def pow(mat, p, target=None):
    """
    If p is a scalar, compute the 'p'th power of each element of the matrix mat,
    otherwise raise each element of the matrix mat to the power given by the
    corresponding element of the matrix p.
    """

    if not target:
        target = mat

    if isinstance(p, CUDAMatrix):
        err_code = _cudamat.apply_pow_matrix(mat.p_mat, p.p_mat, target.p_mat)
    elif isinstance(p, (int, float)):
        err_code = _cudamat.apply_pow(mat.p_mat, ct.c_double(p), target.p_mat)
    else:
        raise ValueError("Value must be of type CUDAMatrix, int, or float.")

    if err_code:
        raise generate_exception(err_code)

    return target


def where(condition_mat, if_mat, else_mat, target=None):
    """
    For each element i, j, store if_math[i, j] in target[i,j] if
    condition_mat[i, j] is True, and else_mat[i, j] otherwise.
    """
    if not target:
        target = condition_mat

    err_code = _cudamat.where(condition_mat.p_mat, if_mat.p_mat,
                              else_mat.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def correlate(mat, kernel, target=None):
    """
    Cross-correlate a matrix with a kernel matrix.
    The kernel matrix is centered over each element of the matrix mat.
    Width and height of the kernel matrix must be an odd integer.
    If a target is not provided, a new matrix is created for storing the result.
    Note that this function cannot operate in-place.
    """
    if not target:
        m = _cudamat.get_leading_dimension(mat.p_mat)
        n = _cudamat.get_nonleading_dimension(mat.p_mat)

        target = empty((m, n))

    err_code = _cudamat.correlate(mat.p_mat, kernel.p_mat, target.p_mat)
    if err_code:
        raise generate_exception(err_code)

    return target


def cuda_sync_threads():
    _cudamat.cuda_sync_threads()


def reformat(array, copy=True):
    """
    Returns array as a float64 array in FORTRAN order.
    If copy is set to False, the array will only be copied if it is not already
    in the correct format.
    """
    return np.array(array, dtype=np.float64, order='F', copy=copy)


def cuda_set_device(dev_id):
    """
    Selects the CUDA device with the given ID.
    """

    err_code = _cudamat.cuda_set_device(ct.c_int(dev_id))
    if err_code:
        raise generate_exception(err_code)


def cublas_init(max_ones=(1024*256)):
    """
    Initialize Cublas.

    'max_ones' is an optional argument that determines the length of
    the largest sum that can be computed using Cublas matrix multiply.
    A larger value causes more memory to be allocated for this purpose.
    """

    err = _cudamat.cublas_init()
    if err:
        raise CUDAMatException('error initializing CUBLAS: (err=%u)' % err)
    CUDAMatrix.ones = empty((max_ones, 1)).assign(1.0)

init = cublas_init


def cublas_shutdown():
    """
    Shut down Cublas.
    """

    CUDAMatrix.ones = 0
    _cudamat.cublas_shutdown()

shutdown = cublas_shutdown
