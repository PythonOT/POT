---
name: Bug report
about: Create a report to help us improve POT
title: ''
labels: bug, help wanted
assignees: ''

---

## Describe the bug
<!-- A clear and concise description of what the bug is. -->

### To Reproduce
Steps to reproduce the behavior:
1. ...
2. 

<!-- If you have error messages or stack traces, please provide it here as well -->

#### Screenshots
<!-- If applicable, add screenshots to help explain your problem. -->

#### Code sample
<!-- Ideally attach a minimal code sample to reproduce the decried issue. 
Minimal means having the shortest code but still preserving the bug. -->

### Expected behavior
<!-- A clear and concise description of what you expected to happen. -->


### Environment (please complete the following information):
- OS (e.g. MacOS, Windows, Linux):
- Python version:
- How was POT installed (source, `pip`, `conda`):
- Build command you used (if compiling from source):
- Only for GPU related bugs:
    - CUDA version:
    - GPU models and configuration:
    - Any other relevant information:

Output of the following code snippet:
```python
import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import ot; print("POT", ot.__version__)
```

### Additional context
<!-- Add any other context about the problem here. -->
