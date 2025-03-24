"""
Finite element code for solving the
1D piecewise linear Galerkin DiffEq (D):
    -(Diff u')' = f  in (a,b)
    u(a) = u(b) = 0  (Dirichlet BC)

1D piecewise quadratic Galerkin DiffEq (CD):
    -(Diff u')' + b u' = f in (a,b)
    u(a) = 0; u(b) = 1
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import cycle


class TestParams:
    K_CONST = 4
    N_DIRICHLET = 2

    MIN_NODES_EXP = 3
    MAX_NODES_EXP = 12

    X_START = 0
    X_END = 1

    @staticmethod
    def diffusion_function(x):
        # half_nodes = len(x) // 2
        # other_half = len(x) - half_nodes
        # return np.concat((np.ones(half_nodes), 2 * np.ones(other_half)))
        return np.ones(len(x))

    @staticmethod
    def u_real(x):
        """
        u(x) = sin(k*pi*x)
        """
        return np.sin(TestParams.K_CONST * math.pi * x)

    @staticmethod
    def source(x):
        """
        f(x) = (k*pi)^2 sin(k*pi*x)
        """
        return TestParams.diffusion_function(x) * (TestParams.K_CONST * math.pi)**2 * TestParams.u_real(x)


def elem_indices(n: int, m: int) -> np.ndarray:
    """
    Create an integer array representing element indices,
    with one element per row

    e.g.: n = 3, m = 3
        [0, 1, 2, 3]
        [3, 4, 5, 6]
        [6, 7, 8, 9]

    :param n: num of elements (rows)
    :param m: 1 - num of subpoints
    :return: n * (m + 1) array
    """
    i_max = n * m

    elem_wo_last_col = np.reshape(np.array(range(i_max), dtype=int), (n, m))
    last_col = np.append(elem_wo_last_col[1:, 0], i_max)

    return np.column_stack((elem_wo_last_col, last_col))


def _impose_neumann_boundary(
    n_neu: int,
    stiff_mat: np.ndarray,
    penalty=1e15,
) -> [np.ndarray, np.ndarray]:
    pass


def impose_boundary(
    n_dir: int,
    boundary_nodes: np.ndarray,
    boundary_vals: np.ndarray,
    stiff_mat: np.ndarray,
    rhs: np.ndarray,
    penalty=1e15,
    neumann=False
) -> [np.ndarray, np.ndarray]:
    if neumann:
        return _impose_neumann_boundary(n_dir, penalty=penalty)

    for idir in range(n_dir):
        iglob = boundary_nodes[idir]
        stiff_mat[iglob, iglob] = penalty
        rhs[iglob] = penalty * boundary_vals[idir]
    return stiff_mat, rhs


def input_data(x_start: float, x_end: float, n_elem: int, polydeg: int) \
        -> [
            int,
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray
        ]:
    """

    :param x_start:
    :param x_end:
    :param n_elem:
    :return:
    """

    # discretize interval [a,b] by
    # (n_elements+1) intervals
    nodes = n_elem * polydeg + 1
    x_coords = np.linspace(x_start, x_end, nodes)

    # define element indices
    elements = elem_indices(n_elem, polydeg)

    # define Dirichlet boundary conditions
    n_dirichlet = TestParams.N_DIRICHLET  # for 1D case
    dirichlet_nodes = np.zeros((n_dirichlet, 1), dtype=int)
    dirichlet_vals = np.zeros((n_dirichlet, 1), dtype=int)
    dirichlet_nodes[0] = 0; dirichlet_nodes[1] = nodes - 1
    dirichlet_vals[0] = 0; dirichlet_vals[1] = 0

    # define Diffusion Coefficient
    diffusion = TestParams.diffusion_function(x_coords)

    # define source function
    k_const = TestParams.K_CONST
    u_real = TestParams.u_real
    source = TestParams.source

    return nodes, elements, x_coords, n_dirichlet, dirichlet_nodes, dirichlet_vals, diffusion, source, u_real


def local_basis(n_elem, x_coords, elems, polydeg=1):
    phi_0 = np.polynomial.Polynomial((1, -1), domain=[0, 1], window=[0, 1])
    phi_1 = np.polynomial.Polynomial((0, 1), domain=[0, 1], window=[0, 1])

    if polydeg == 2:
        quadratic_bubble = np.polynomial.Polynomial((0, 1, -1), domain=[0, 1], window=[0, 1])
        p2_basis = phi_0 - quadratic_bubble, phi_1 + quadratic_bubble
        return p2_basis

    p1_basis = phi_0, phi_1
    return p1_basis


def build_stiffness_matrix(x_coords, basis_coeff, diff):
    # basis derivatives
    phi_0, phi_1 = basis_coeff
    dphi_0 = phi_0.deriv()(0.5)
    dphi_1 = phi_1.deriv()(0.5)

    c0 = dphi_0 ** 2
    c1 = dphi_0 * dphi_1

    # finite differences (h_i)
    x_diff = np.diff(x_coords)
    x_mid = x_coords[:-1] + x_diff/2

    # compute diffusion midpoints
    diff_mid = (diff[1:] + diff[:-1])/2
    integral_factors = np.multiply(x_diff**(-1), diff_mid)

    # handle edge case
    integral_factors = np.append(integral_factors, [0])

    # compute main diagonal A[i,i]
    # initialize stiff_mat
    diag = c0 * (integral_factors + np.roll(integral_factors, 1))
    stiff_mat = np.diag(diag)

    # compute off-diagonal A[i,i+1]
    off_diag = c1 * integral_factors[:-1]
    np.fill_diagonal(stiff_mat[1:, :], off_diag)
    np.fill_diagonal(stiff_mat[:, 1:], off_diag)

    return stiff_mat


def gauss_quadrature(f, dom, n: int):
    """

    :param f: integrand
    :param dom: domain
    :param n: number of sample points (>= 1)
    :return: integral
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    pass


def build_load_vector(x_coords, source, basis_coeff):
    # finite differences (h_i)
    x_diff = np.diff(x_coords)

    phi_0, phi_1 = basis_coeff
    p_right = phi_0(0.5)
    p_left = phi_1(0.5)

    # compute source midpoints
    source_pts = source(x_coords)
    source_mid = (source_pts[1:] + source_pts[:-1])/2

    common_factor = np.append(x_diff * source_mid, [0])
    right_integral = p_right * common_factor
    left_integral = p_left * np.roll(common_factor, 1)

    load_vector = left_integral + right_integral

    return load_vector


def galerkin(n_elem: int, x_start=0.0, x_end=1.0, polydeg=1):
    nodes, elems, x_coords, n_dirichlet, dirichlet_nodes, dirichlet_vals, diffusion, source, u_real \
        = input_data(x_start, x_end, n_elem, polydeg)
    basis_coeff = local_basis(n_elem, x_coords, elems)

    stiff_mat = build_stiffness_matrix(x_coords, basis_coeff, diffusion)
    load_vect = build_load_vector(x_coords=x_coords, source=source, basis_coeff=basis_coeff)

    stiff_mat, load_vect = impose_boundary(
        n_dir=n_dirichlet,
        boundary_nodes=dirichlet_nodes,
        boundary_vals=dirichlet_vals,
        stiff_mat=stiff_mat,
        rhs=load_vect)

    u_approx = np.linalg.solve(stiff_mat, load_vect)

    return x_coords, u_approx, u_real


def calc_l2err(x, f_approx, f_real, nodes_only=False):
    if nodes_only:
        f_real_vals = f_real(x)
        return np.linalg.norm(f_real_vals - f_approx)

    fine_x = np.linspace(x[0], x[-1], 10000)
    f_real_vals = f_real(fine_x)
    f_approx_vals = np.interp(fine_x, x, f_approx)

    error_squared = (f_real_vals - f_approx_vals) ** 2
    integral = np.trapezoid(error_squared, fine_x)

    return np.sqrt(integral)


def convergence_test():
    n_elems = [2**i for i in range(TestParams.MIN_NODES_EXP, TestParams.MAX_NODES_EXP + 1)]
    x_start = TestParams.X_START
    x_end = TestParams.X_END

    residuals = []

    for n_elem in n_elems:
        x_coords, u_approx, u_real = galerkin(n_elem, x_start, x_end)
        plt.plot(x_coords, u_approx)

        residuals.append(calc_l2err(x_coords, u_approx, u_real))

    x_fine = np.linspace(x_start, x_end, 10 ** 4)
    plt.plot(x_fine, TestParams.u_real(x_fine))
    plt.ylim((-1.25, 1.25))

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("u_real and u_approx")
    plt.show()

    residuals = np.array(residuals)
    plt.plot(np.log(residuals))
    plt.show()

    ratios = np.divide(residuals[:-1], residuals[1:])
    print(ratios)
    plt.plot(ratios)
    plt.show()

    return


def main():
    n_elems = [2 ** i for i in range(TestParams.MIN_NODES_EXP, TestParams.MAX_NODES_EXP + 1)]
    x_start = TestParams.X_START
    x_end = TestParams.X_END

    residuals = []

    # Define marker cycle
    markers = cycle(['o', 's', '^', 'x', 'D', 'v', 'P', '*'])

    for n_elem in n_elems:
        x_coords, u_approx, u_real = galerkin(n_elem, x_start, x_end)
        plt.plot(x_coords, u_approx, marker=next(markers), linestyle='--', markersize=3)

        residuals.append(calc_l2err(x_coords, u_approx, u_real))

    # x_fine = np.linspace(x_start, x_end, 10 ** 4)
    # plt.ylim((-1.25, 1.25))

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("u_real and u_approx")
    plt.show()


if __name__ == '__main__':
    convergence_test()

