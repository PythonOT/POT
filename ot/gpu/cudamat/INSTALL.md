Prerequisites
-------------

cudamat needs the following to be installed first:

* Python 2.x or 3.x and numpy
* The CUDA SDK
* nose for running the tests (optional)

Installation
------------

Once you have installed the prerequisites and downloaded cudamat, switch to the
cudamat directory and run either of the following commands to install it:

```bash
# a) Install for your user:
python setup.py install --user
# b) Install for your user, but with pip:
pip install --user .
# c) Install system-wide:
sudo python setup.py install
# d) Install system-wide, but with pip:
sudo pip install .
```

If your Nvidia GPU supports a higher Compute Capability than the default one of
your CUDA toolkit, you can set the `NVCCFLAGS` environment variable when
installing cudamat to compile it for your architecture. For example, to install
for your user for a GTX 780 Ti (Compute Capability 3.5), you would run:

```bash
NVCCFLAGS=-arch=sm_35 python setup.py install --user
```

To compile for both Compute Capability 2.0 and 3.5, you would run:

```bash
NVCCFLAGS="-gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35" ...
```

Testing
-------

To test your setup, run the included unit tests and optionally the benchmark:

```bash
cd test  # so it doesn't try importing cudamat from the source directory
# Run tests
nosetests
# Run benchmark
python ../examples/bench_cudamat.py
```
