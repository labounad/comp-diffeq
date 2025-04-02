import numpy as np
from fem.fem1d import (
    Fem1dInput,
    _elem_indices,
    accumulate_by_index,
    build_load_vector,
    build_stiffness_matrix,
    calc_l2err,
    local_basis,
    galerkin
)
from scipy.stats import linregress


# ------------------------
# elem_indices
# ------------------------

def test_elem_indices_linear():
    result = _elem_indices(3, 1)
    expected = np.array([[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(result, expected)


def test_elem_indices_quadratic():
    result = _elem_indices(2, 2)
    expected = np.array([[0, 1, 2], [2, 3, 4]])
    np.testing.assert_array_equal(result, expected)


# ------------------------
# accumulate_by_index
# ------------------------

def test_accumulate_by_index_basic():
    values = np.array([[1, 2, 3], [4, 5, 6]])
    indices = np.array([[0, 1, 2], [2, 3, 4]])
    result = accumulate_by_index(values, indices)
    expected = np.array([1, 2, 7, 5, 6])
    np.testing.assert_array_equal(result, expected)


# ------------------------
# build_load_vector
# ------------------------

def test_build_load_vector_linear_three_elements():
    fem_input = Fem1dInput(domain=(0, 1), n_elems=3, polydeg=1)
    source = lambda x: np.ones_like(x)
    basis = local_basis(1)

    load_vector = build_load_vector(fem_input.x_coords, 1, fem_input.elements, source, basis)
    expected = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    np.testing.assert_allclose(load_vector, expected, rtol=1e-8)


def test_build_load_vector_quadratic_two_elements():
    fem_input = Fem1dInput(domain=(0, 1), n_elems=2, polydeg=2)
    source = lambda x: np.ones_like(x)
    basis = local_basis(2)

    load_vector = build_load_vector(fem_input.x_coords, 2, fem_input.elements, source, basis)
    expected = np.array([1 / 12, 1 / 3, 1 / 6, 1 / 3, 1 / 12])
    np.testing.assert_allclose(load_vector, expected, rtol=1e-8)


# ------------------------
# build_stiffness_matrix
# ------------------------

def test_build_stiffness_matrix_linear_three_elements():
    fem_input = Fem1dInput(domain=(0, 1), n_elems=3, polydeg=1)
    diffusion = lambda x: np.ones_like(x)
    basis = local_basis(1)

    stiff_mat = build_stiffness_matrix(fem_input.x_coords, 1, fem_input.elements, basis, diffusion)

    expected = np.array([
        [3, -3, 0, 0],
        [-3, 6, -3, 0],
        [0, -3, 6, -3],
        [0, 0, -3, 3]
    ])
    np.testing.assert_allclose(stiff_mat, expected, atol=1e-8)


def test_build_stiffness_matrix_quadratic_two_elements():
    fem_input = Fem1dInput(domain=(0, 1), n_elems=2, polydeg=2)
    diffusion = lambda x: np.ones_like(x)
    basis = local_basis(2)

    stiff_mat = build_stiffness_matrix(fem_input.x_coords, 2, fem_input.elements, basis, diffusion)

    expected = np.array([
        [14 / 3, -16 / 3, 2 / 3, 0, 0],
        [-16 / 3, 32 / 3, -16 / 3, 0, 0],
        [2 / 3, -16 / 3, 28 / 3, -16 / 3, 2 / 3],
        [0, 0, -16 / 3, 32 / 3, -16 / 3],
        [0, 0, 2 / 3, -16 / 3, 14 / 3]
    ])
    np.testing.assert_allclose(stiff_mat, expected, atol=1e-8)


def test_build_stiffness_matrix_quadratic_five_elements():
    fem_input = Fem1dInput(domain=(0, 1), n_elems=5, polydeg=2)
    diffusion = lambda x: np.ones_like(x)
    basis = local_basis(2)

    stiff_mat = build_stiffness_matrix(fem_input.x_coords, 2, fem_input.elements, basis, diffusion)

    np.testing.assert_equal(stiff_mat.shape, (11, 11))
    np.testing.assert_allclose(stiff_mat, stiff_mat.T, atol=1e-8)  # symmetry test


# ------------------------
# comprehensive convergence test
# ------------------------

def test_convergence_linear_with_known_solution():
    k = 7
    u_real = lambda x: np.sin(k * np.pi * x)
    domain = (0, 1)
    polydeg = 1
    n_elems = [2**i for i in range(4, 9)]  # 16 to 256 elements

    errors, hs = [], []
    for n_elem in n_elems:
        fem_input = Fem1dInput(domain=domain, n_elems=n_elem, polydeg=polydeg)
        u_approx_func = galerkin(fem_input, return_function=True)
        error = calc_l2err(u_approx_func, u_real, fem_input)
        h = (domain[1] - domain[0]) / n_elem
        errors.append(error)
        hs.append(h)

    slope, _, *_ = linregress(np.log(hs), np.log(errors))
    assert 1.9 <= slope <= 2.1, f"Linear basis: Expected O(h^2) convergence, got slope = {slope:.4f}"


def test_convergence_quadratic_with_known_solution():
    k = 7
    u_real = lambda x: np.sin(k * np.pi * x)
    domain = (0, 1)
    polydeg = 2
    n_elems = [2**i for i in range(3, 8)]  # 8 to 128 elements

    errors, hs = [], []
    for n_elem in n_elems:
        fem_input = Fem1dInput(domain=domain, n_elems=n_elem, polydeg=polydeg)
        u_approx_func = galerkin(fem_input, return_function=True)
        error = calc_l2err(u_approx_func, u_real, fem_input)
        h = (domain[1] - domain[0]) / n_elem
        errors.append(error)
        hs.append(h)

    slope, _, *_ = linregress(np.log(hs), np.log(errors))
    assert 2.9 <= slope <= 3.1, f"Quadratic basis: Expected O(h^3) convergence, got slope = {slope:.4f}"
