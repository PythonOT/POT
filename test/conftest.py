# -*- coding: utf-8 -*-

# Configuration file for pytest

# License: MIT License

import pytest
from ot.backend import jax
from ot.backend import get_backend_list
import functools

if jax:
    from jax.config import config
    config.update("jax_enable_x64", True)

backend_list = get_backend_list()


@pytest.fixture(params=backend_list)
def nx(request):
    backend = request.param

    yield backend


def skip_arg(arg, value, reason=None, getter=lambda x: x):
    if reason is None:
        reason = f"Param {arg} should be skipped for value {value}"

    def wrapper(function):

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if arg in kwargs.keys() and getter(kwargs[arg]) == value:
                pytest.skip(reason)
            return function(*args, **kwargs)

        return wrapped

    return wrapper


def pytest_configure(config):
    pytest.skip_arg = skip_arg
    pytest.skip_backend = functools.partial(skip_arg, "nx", getter=str)
