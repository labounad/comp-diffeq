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
from fem.fem1d import _elem_indices


def test_elem_indices():
    result = _elem_indices(3, 1)
    expected = np.array([[0, 1],
                         [1, 2],
                         [2, 3]])
    np.testing.assert_array_equal(result, expected)
```


## 🚀 Using the FEM Command Line Interface (CLI)

The `fem` CLI allows easy numerical solutions of differential equations using the Finite Element Method (FEM). You can specify PDE parameters, source terms, and other details directly from the command line.

---

### 📥 Installation and Setup

Ensure your environment is correctly configured:

```bash
conda env create -f environment.yml
conda activate comp-diffeq
```

---

## 🖥️ CLI Usage

Use the CLI directly from your project's root directory:

```bash
python -m cli.fem solve <n_dim> <source_function> [OPTIONS]
```

**Example:**

Solve a simple 1D PDE with default settings (quadratic basis, 1000 elements):

```bash
python -m cli.fem solve 1 "sin(7*pi*x)"
```

Specify custom options:

```bash
python -m cli.fem solve 1 "x**2 + 3*x + 1" --diffusion "1 + x**2" --polydeg 1 --n-elems 500
```

---

### 📖 Command and Option Details

**Basic syntax:**

```
python -m cli.fem solve <n_dim> <source_function> [--diffusion FUNC] [--polydeg INT] [--n-elems INT]
```

**Arguments:**

- `<n_dim>` *(required)*: Dimension of the PDE problem (currently only `1` is supported).
- `<source_function>` *(required)*: Source function (right-hand side of PDE), expressed as a string in terms of `x`.

**Options:**

| Option           | Default  | Description                                           | Example                       |
|------------------|----------|-------------------------------------------------------|-------------------------------|
| `--diffusion`    | `"1"`    | Diffusion coefficient as a function of `x`.           | `"1 + x**2"`                  |
| `--polydeg`      | `2`      | Degree of polynomial basis functions (`1` or `2`).    | `--polydeg 1`                 |
| `--n-elems`      | `1000`   | Number of finite elements.                            | `--n-elems 500`               |

---

### 🧮 Examples

**Solve a 1D PDE with quadratic basis (default):**

```bash
python -m cli.fem solve 1 "sin(5*pi*x)"
```

**Solve with linear basis and custom diffusion:**

```bash
python -m cli.fem solve 1 "exp(x)" --diffusion "2 + sin(x)" --polydeg 1
```

---

### ✅ Interpreting Output

Upon solving, the CLI provides the solution at the midpoint (`x=0.5`) as an immediate check:

```
Solution at midpoint (x=0.5): u(0.5) = 0.12345
```

---

### ⚠️ Limitations and Planned Improvements

- **Current support**: Only one-dimensional PDEs.
- **Future updates**: Multi-dimensional PDE support, customizable boundary conditions, mesh refinement options, and more PDE types.

---

### 🧪 Running Tests

Verify the CLI's functionality using `pytest`:

```bash
pytest tests/test_cli.py
```

---

### 🚧 Troubleshooting

- **"ModuleNotFoundError"**: Ensure your `comp-diffeq` conda environment is activated.
- **"Sympy Parsing Error"**: Check syntax in your function strings (e.g., `"sin(pi*x)"` instead of `"sin(pi x)"`).




