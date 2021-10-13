# -*- coding: utf-8 -*-

# Configuration file for pytest

# License: MIT License

import pytest
from ot.backend import jax

if jax:
    from jax.config import config


@pytest.fixture
def nx(request):
    backend = request.param
    if backend.__name__ == "jax":
        config.update("jax_enable_x64", True)

    yield backend

    if backend.__name__ == "jax":
        config.update("jax_enable_x64", False)
