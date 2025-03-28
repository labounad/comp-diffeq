# comp-diffeq

## ⚙️ Environment Setup (Conda)

Follow these instructions to exactly recreate the development environment for this project:

### Step 1: Install Conda

If you haven't installed Conda yet, you can download and install Miniconda here:

- [Miniconda Download](https://docs.conda.io/en/latest/miniconda.html)

### Step 2: Clone the Repository

```bash
git clone https://github.com/labounad/comp-diffeq.git
cd comp-diffeq
```

### Step 3: Create the Conda Environment

Create the environment from the provided YAML file (`environment.yml`):

```bash
conda env create -f environment.yml
```

### Step 4: Activate the Environment

Activate the newly created environment:

```bash
conda activate comp-diffeq
```

Your environment is now ready and matches the original development setup.


## 🧪 Testing with pytest

This project uses the [pytest](https://docs.pytest.org/) framework for automated testing.

### How to run tests:

Activate the conda environment first (if not activated already):

```bash
conda activate comp-diffeq
```

Run all tests using:

```bash
pytest
```

For more detailed output (verbose mode):

```bash
pytest -v
```

### Writing new tests:

- Create new test files under the `tests/` directory.
- Name test files starting with `test_`.
- Define each test function starting clearly with `test_`.

Example:

```python
# tests/test_fem1d.py

import numpy as np
from fem.fem1d import elem_indices

def test_elem_indices():
    result = elem_indices(3, 1)
    expected = np.array([[0, 1],
                         [1, 2],
                         [2, 3]])
    np.testing.assert_array_equal(result, expected)
```


