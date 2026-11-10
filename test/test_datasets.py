"""Tests for module datasets"""

# License: MIT License

import pytest

from ot.datasets import make_data_classif


def test_make_data_classif_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        make_data_classif("not_a_dataset", 10)
