Version 0.3
- Added equality testing. Contributed by Ryan P. Adams.
- Added flag that determines whether threads are synced after each call.
- Added direct blas calls for two elementwise operations. Contributed by Vincent Vanhoucke.
- Added the set_selected_columns. Contributed by Tijmen Tieleman.
- Added tanh, abs, and log_1_plus_exp. Contributed by Ilya Sutskever.
- Some bug fixes. Contributed by Jonathan Taylor.
- Added the select_columns method. Code contributed by Tijmen Tieleman.
- on_device should now work as intended.
- allocate_device_memory now returns an error when cublasAlloc fails.
- Fixed bug in max that showed up when an entire column was negative.
- Fixed bug in activation computations in examples/rbm_numpy.py.
- Added get_col_slice and set_col_slice methods.
- Added init and shutdown methods to shorten cublas_init and cublas_shutdown.
- Added bound checking to the various slicing methods.
- Fixed problem with pow and negative numbers.
- Added support for matrix powers in pow.

Version 0.2
- Methods add, subtract, mult, divide can now take scalars as well as instances of CUDAMatrix.
- Deprecated add_scalar, mult_by_scalar, div_by_scalar.
- Methods now return target or self to make chaining operations easier.
- Added asarray method.
- Added transpose method.
- Added sqrt and pow functions.
- Added the sigmoid method to the module level.
- Added add_row_vec.
- Added empty. Now when you don't provide a target or pre-allocated temporary storage cudamat methods will not take up CPU RAM or transfer anything between the CPU and GPU.
- Added get_row_slice and set_row_slice.
- Added less_than_scalar, greater_than, greater_than_scalar.
- Added max (axis=1 is currently not supported.)

Version 0.1.5
- Added shape attribute and reshape method.

Version 0.1
- Most methods now throw python exceptions instead of exiting after encountering an error.
- The CUDAMatrix constructor now automatically converts ndarray objects to float32 in FORTRAN order.
- Renamed scalar_mult to mult_by_scalar and scalar_div to div_by_scalar.
- Added log and exp functions.
- Removed add_row_sums and sum_rows.
