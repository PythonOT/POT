# -*- coding: utf-8 -*-

# Configuration file for pytest

# License: MIT License

import functools
import os
import pytest

from ot.backend import get_backend_list, jax, tf


if jax:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    from jax import config
    config.update("jax_enable_x64", True)

if tf:
    # make sure TF doesn't allocate entire GPU
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception:
            pass

    # allow numpy API for TF
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()


backend_list = get_backend_list()


@pytest.fixture(params=backend_list)
def nx(request):
    backend = request.param

    yield backend


def skip_arg(arg, value, reason=None, getter=lambda x: x):
    if isinstance(arg, (tuple, list)):
        n = len(arg)
    else:
        arg = (arg, )
        n = 1
    if n != 1 and isinstance(value, (tuple, list)):
        pass
    else:
        value = (value, )
    if isinstance(getter, (tuple, list)):
        pass
    else:
        getter = [getter] * n

    if reason is None:
        reason = f"Param {arg} should be skipped for value {value}"

    def wrapper(function):

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if all(
                arg[i] in kwargs.keys() and getter[i](kwargs[arg[i]]) == value[i]
                for i in range(n)
            ):
                pytest.skip(reason)
            return function(*args, **kwargs)

        return wrapped

    return wrapper


def pytest_configure(config):
    pytest.skip_arg = skip_arg
    pytest.skip_backend = functools.partial(skip_arg, "nx", getter=str)
