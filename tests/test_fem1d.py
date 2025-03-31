import numpy as np
from numpy.polynomial import Polynomial
from fem.fem1d import (
    impose_boundary,
    elem_indices,
    accumulate_by_index,
    build_load_vector,
    build_stiffness_matrix,
    calc_l2err
)

# ------------------------
# Helper functions
# ------------------------


def local_basis(polydeg=1):
    if polydeg == 1:
        return [
            Polynomial([0.5, -0.5]),  # (1 - ξ)/2
            Polynomial([0.5,  0.5])   # (1 + ξ)/2
        ]
    elif polydeg == 2:
        return [
            Polynomial([0, -0.5, 0.5]),  # ξ(ξ - 1)/2
            Polynomial([1.0, 0.0, -1.0]),  # 1 - ξ²
            Polynomial([0, 0.5, 0.5])  # ξ(ξ + 1)/2
        ]
    else:
        raise ValueError("Only degrees 1 and 2 are supported.")


def prep_gauss_quadrature(f, dom, nodes):
    J = (dom[1:] - dom[:-1]) / 2
    midpoints = (dom[1:] + dom[:-1]) / 2
    x_k = J[:, None] * nodes + midpoints[:, None]
    f_vals = f(x_k)
    return f_vals * J[:, None]


# ------------------------
# elem_indices
# ------------------------


def test_elem_indices_linear():
    result = elem_indices(3, 1)
    expected = np.array([[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(result, expected)


def test_elem_indices_quadratic():
    result = elem_indices(2, 2)
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
    x_coords = np.linspace(0, 1, 4)  # 3 elements → 4 nodes
    elems = elem_indices(3, 1)
    source = lambda x: np.ones_like(x)
    basis = local_basis(1)

    load_vector = build_load_vector(x_coords, 1, elems, source, basis)
    expected = np.array([1/6, 1/3, 1/3, 1/6])
    np.testing.assert_allclose(load_vector, expected, rtol=1e-8)


def test_build_load_vector_quadratic_two_elements():
    x_coords = np.linspace(0, 1, 5)  # 2 quadratic elements → 5 nodes
    elems = elem_indices(2, 2)
    source = lambda x: np.ones_like(x)
    basis = local_basis(2)

    load_vector = build_load_vector(x_coords, 2, elems, source, basis)
    expected = np.array([1/12, 1/3, 1/6, 1/3, 1/12])
    np.testing.assert_allclose(load_vector, expected, rtol=1e-8)


# ------------------------
# build_stiffness_matrix
# ------------------------


def test_build_stiffness_matrix_linear_three_elements():
    x_coords = np.linspace(0, 1, 4)  # 3 elements of size h = 1/3
    elems = elem_indices(3, 1)
    diffusion = lambda x: np.ones_like(x)
    basis = local_basis(1)

    stiff_mat = build_stiffness_matrix(x_coords, 1, elems, basis, diffusion)

    expected = np.array([
        [ 3, -3,  0,  0],
        [-3,  6, -3,  0],
        [ 0, -3,  6, -3],
        [ 0,  0, -3,  3]
    ])
    np.testing.assert_allclose(stiff_mat, expected, atol=1e-8)


def test_build_stiffness_matrix_quadratic_two_elements():
    x_coords = np.linspace(0, 1, 5)  # 2 quadratic elements → 5 nodes
    elems = elem_indices(2, 2)
    diffusion = lambda x: np.ones_like(x)
    basis = local_basis(2)

    stiff_mat = build_stiffness_matrix(x_coords, 2, elems, basis, diffusion)

    expected = np.array([
        [14 / 3, -16 / 3, 2 / 3, 0, 0],
        [-16 / 3, 32 / 3, -16 / 3, 0, 0],
        [2 / 3, -16 / 3, 28 / 3, -16 / 3, 2 / 3],
        [0, 0, -16 / 3, 32 / 3, -16 / 3],
        [0, 0, 2 / 3, -16 / 3, 14 / 3]
    ])

    np.testing.assert_allclose(stiff_mat, expected, atol=1e-8)


def test_build_stiffness_matrix_quadratic_five_elements():
    x_coords = np.linspace(0, 1, 11)  # 5 quadratic elements → 11 nodes
    elems = elem_indices(5, 2)
    diffusion = lambda x: np.ones_like(x)
    basis = local_basis(2)

    stiff_mat = build_stiffness_matrix(x_coords, 2, elems, basis, diffusion)

    expected = np.array([
        [11.66666667, -13.33333333, 1.66666667, 0., 0., 0., 0., 0., 0., 0., 0.],
        [-13.33333333, 26.66666667, -13.33333333, 0., 0., 0., 0., 0., 0., 0., 0.],
        [1.66666667, -13.33333333, 23.33333333, -13.33333333, 1.66666667, 0., 0., 0., 0., 0., 0.],
        [0., 0., -13.33333333, 26.66666667, -13.33333333, 0., 0., 0., 0., 0., 0.],
        [0., 0., 1.66666667, -13.33333333, 23.33333333, -13.33333333, 1.66666667, 0., 0., 0., 0.],
        [0., 0., 0., 0., -13.33333333, 26.66666667, -13.33333333, 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.66666667, -13.33333333, 23.33333333, -13.33333333, 1.66666667, 0., 0.],
        [0., 0., 0., 0., 0., 0., -13.33333333, 26.66666667, -13.33333333, 0., 0.],
        [0., 0., 0., 0., 0., 0., 1.66666667, -13.33333333, 23.33333333, -13.33333333, 1.66666667],
        [0., 0., 0., 0., 0., 0., 0., 0., -13.33333333, 26.66666667, -13.33333333],
        [0., 0., 0., 0., 0., 0., 0., 0., 1.66666667, -13.33333333, 11.66666667]
    ])

    np.testing.assert_allclose(stiff_mat, expected, rtol=1e-6)


# ------------------------
# comprehensive testing
# ------------------------


def test_quadratic_convergence_with_known_solution():
    from scipy.stats import linregress

    # Define a known solution and source term
    k = 7
    u_real = lambda x: np.sin(k * np.pi * x)
    source = lambda x: (k * np.pi) ** 2 * np.sin(k * np.pi * x)

    # Override diffusion and input parameters for this test only
    diffusion = lambda x: np.ones_like(x)
    polydeg = 2  # Quadratic
    x_start, x_end = 0, 1

    # Range of element counts (non-trivial, varying h)
    n_elems = [2**i for i in range(3, 8)]  # i.e., 8 to 128 elements

    errors = []
    hs = []

    for n_elem in n_elems:
        x_coords = np.linspace(x_start, x_end, n_elem * polydeg + 1)
        elems = elem_indices(n_elem, polydeg)
        basis = local_basis(polydeg)

        stiff = build_stiffness_matrix(x_coords, polydeg, elems, basis, diffusion)
        load = build_load_vector(x_coords, polydeg, elems, source, basis)

        # Apply boundary conditions u(0) = u(1) = 0
        stiff, load = impose_boundary(
            n_dir=2,
            boundary_nodes=np.array([0, len(x_coords)-1]),
            boundary_vals=np.array([0, 0]),
            stiff_mat=stiff,
            rhs=load
        )

        u_approx = np.linalg.solve(stiff, load)
        err = calc_l2err(x_coords, u_approx, u_real)
        h = (x_end - x_start) / n_elem

        errors.append(err)
        hs.append(h)

    # Fit log(error) = p log(h) + C
    log_h = np.log(hs)
    log_e = np.log(errors)
    slope, intercept, *_ = linregress(log_h, log_e)

    print(f"Estimated convergence rate: {slope:.4f}")
    assert 1.9 <= -slope <= 2.1, f"Expected O(h^2) convergence, got slope = {-slope:.4f}"
