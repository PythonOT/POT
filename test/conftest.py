# -*- coding: utf-8 -*-

# Configuration file for pytest

# License: MIT License

import pytest
from ot.backend import jax
from ot.backend import get_backend_list
import functools

if jax:
    from jax.config import config

backend_list = get_backend_list()


@pytest.fixture(params=backend_list)
def nx(request):
    backend = request.param
    if backend.__name__ == "jax":
        config.update("jax_enable_x64", True)

    yield backend

    if backend.__name__ == "jax":
        config.update("jax_enable_x64", False)


def skip_backend(backend_to_skip, /, *, reason=None):
    def wrapper(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if "nx" not in kwargs.keys():
                raise TypeError("Cannot call the skip_backend decorator if the backend is not called")
            else:
                current_backend = kwargs["nx"].__name__
                if current_backend == backend_to_skip:
                    nonlocal reason
                    if reason is None:
                        reason = f"{current_backend} not supported for this function"
                    pytest.skip(reason)
                return function(*args, **kwargs)
        return wrapped
    return wrapper


def pytest_configure(config):
    pytest.skip_backend = skip_backend
