"""Tests for helpers functions """

# Author: Remi Flamary <remi.flamary@polytechnique.edu>
#
# License: MIT License

import os
import sys

sys.path.append(os.path.join("ot", "helpers"))

from openmp_helpers import get_openmp_flag, check_openmp_support  # noqa
from pre_build_helpers import _get_compiler, compile_test_program  # noqa


def test_helpers():

    compiler = _get_compiler()

    get_openmp_flag(compiler)

    s = '#include <stdio.h>\n#include <stdlib.h>\n\nint main(void) {\n\tprintf("Hello world!\\n");\n\treturn 0;\n}'
    output, _ = compile_test_program(s)
    assert len(output) == 1 and output[0] == "Hello world!"

    check_openmp_support()
