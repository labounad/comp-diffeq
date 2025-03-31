"""
Finite element code for solving the
1D piecewise linear Galerkin DiffEq (D):
    -(Diff u')' = f  in (a,b)
    u(a) = u(b) = 0  (Dirichlet BC)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import cycle


class TestParams:
    K_CONST = 7
    N_DIRICHLET = 2

    MIN_NODES_EXP = 3
    MAX_NODES_EXP = 12

    X_START = 0
    X_END = 1

    @staticmethod
    def diffusion_function(x):
        # half_nodes = len(x) // 2
        # other_half = len(x) - half_nodes
        # return np.concatenate((np.ones(half_nodes), 2 * np.ones(other_half)))
        return np.ones_like(x)

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
        return TestParams.diffusion_function(x) * (TestParams.K_CONST * math.pi) ** 2 * TestParams.u_real(x)


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
        return _impose_neumann_boundary(n_dir, penalty=penalty)  # DEFUNCT

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
    dirichlet_nodes[0] = 0
    dirichlet_nodes[1] = nodes - 1
    dirichlet_vals[0] = 0
    dirichlet_vals[1] = 0

    # define Diffusion Coefficient
    diffusion = TestParams.diffusion_function

    # define source function
    k_const = TestParams.K_CONST
    u_real = TestParams.u_real
    source = TestParams.source

    return nodes, elements, x_coords, n_dirichlet, dirichlet_nodes, dirichlet_vals, diffusion, source, u_real


def local_basis(polydeg=1):
    bases = {
        1: [np.polynomial.Polynomial((0.5, -0.5), domain=[-1, 1], window=[-1, 1]),  # 1-x
            np.polynomial.Polynomial((0.5, 0.5), domain=[-1, 1], window=[-1, 1])  # x
            ],
        2: [np.polynomial.Polynomial((0, -0.5, 0.5), domain=[-1, 1], window=[-1, 1]),  # -0.5x(1-x)
            np.polynomial.Polynomial((1, 0, -1), domain=[-1, 1], window=[-1, 1]),  # 1-x^2
            np.polynomial.Polynomial((0, 0.5, 0.5), domain=[-1, 1], window=[-1, 1])  # 0.5x(1+x)
            ]
    }

    return bases[polydeg]


def derive_basis(basis):
    return [p.deriv() for p in basis]


def prep_gauss_quadrature(f, dom, nodes):
    """
    Preparing a function f for num integration via gaussian quadrature.

    :param f: integrand
    :param dom: endpoints of domains of integration
    :param nodes: sampling points in interval [-1,1]
    :return: array of points where ith row is
            |T_i| * [f(T_i y_0)  f(T_i y_1),  ...,  f(T_i y_n-1)],
            where
            - |T_i| = (dom_i - dom_i-1) / 2 is the Jacobian of the transform and
            - f(T_i y_j) is the image of the jth node under the
                affine transformation T_i: [-1,1] -> dom_i
    """
    # apply affine transformation from [-1,1] into domains
    transformed_nodes = (np.array((0.5 * np.array([np.diff(dom)])).T @ [nodes + 1]) + dom[:-1][:, np.newaxis])

    return np.diff(dom)[:, np.newaxis] / 2 * f(transformed_nodes)


def accumulate_by_index(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Accumulates values into a 1D array based on given indices.

    Args:
        values (np.ndarray): Array of shape (N, M) containing values to accumulate.
        indices (np.ndarray): Array of shape (N, M) containing integer indices where values should be accumulated.

    Returns:
        np.ndarray: 1D array where each position k contains the
        sum of all entries values[i, j] such that indices[i, j] == k.
    """
    assert values.shape == indices.shape, "values and indices must have the same shape"

    flat_indices = indices.ravel()
    flat_values = values.ravel()

    result_length = flat_indices.max() + 1
    result = np.bincount(flat_indices, weights=flat_values, minlength=result_length)

    return result


def build_stiffness_matrix(x_coords, d, elems, basis, diffusion):
    n_nodes = len(x_coords)
    stiff_mat = np.zeros((n_nodes, n_nodes))

    n_leggauss = d
    leggauss_nodes, leggauss_weights = np.polynomial.legendre.leggauss(n_leggauss)

    # Basis derivatives w.r.t. reference coord ξ
    basis_deriv_vals = np.array([
        p.deriv()(leggauss_nodes) for p in basis  # shape (n_basis, n_quad)
    ])

    for elem in elems:
        elem_x = x_coords[elem]
        x_start, x_end = elem_x[0], elem_x[-1]
        J = (x_end - x_start) / 2
        inv_J = 1 / J

        # Transform Gauss points to physical coordinates
        phys_pts = J * leggauss_nodes + (x_start + x_end) / 2
        weighted_diffusion_vals = diffusion(phys_pts) * leggauss_weights  # shape (n_quad,)

        # Chain rule: dφ/dx = dφ/dξ * (1 / J)
        # basis_phys_derivs = basis_deriv_vals * inv_J  # shape (n_basis, n_quad)

        # Form local stiffness matrix using weighted_D_vals quadrature
        local_stiffness = (basis_deriv_vals * inv_J) @ np.diag(weighted_diffusion_vals) @ basis_deriv_vals.T  # shape (n_basis, n_basis)
        stiff_mat[np.ix_(elem, elem)] += local_stiffness

    return stiff_mat


def build_load_vector(x_coords, d, elems, source, basis):
    element_bounds = np.append(x_coords[elems[:, 0]], x_coords[-1])
    n_leggauss = d // 2 + 1
    leggauss_nodes, leggauss_weights = np.polynomial.legendre.leggauss(n_leggauss)

    basis_vals = np.array([p(leggauss_nodes) for p in basis])  # (n_basis, n_quad)
    weighted_basis_vals = basis_vals * leggauss_weights  # (n_basis, n_quad)

    source_component = prep_gauss_quadrature(source, element_bounds, leggauss_nodes)

    integral_matrix = source_component @ weighted_basis_vals.T

    return accumulate_by_index(integral_matrix, elems)


def galerkin(n_elem: int, x_start=0.0, x_end=1.0, polydeg=1):
    nodes, elems, x_coords, n_dirichlet, dirichlet_nodes, dirichlet_vals, diffusion, source, u_real \
        = input_data(x_start, x_end, n_elem, polydeg)
    basis_coeff = local_basis(polydeg=polydeg)

    stiff_mat = build_stiffness_matrix(x_coords, polydeg, elems, basis_coeff, diffusion)
    load_vect = build_load_vector(x_coords=x_coords, d=polydeg, elems=elems, source=source, basis=basis_coeff)

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
    integral = np.trapz(error_squared, fine_x)

    return np.sqrt(integral)


def convergence_test():
    n_elems = [2 ** i for i in range(TestParams.MIN_NODES_EXP, TestParams.MAX_NODES_EXP + 1)]
    x_start = TestParams.X_START
    x_end = TestParams.X_END

    residuals = []

    for n_elem in n_elems:
        x_coords, u_approx, u_real = galerkin(n_elem, x_start, x_end, polydeg=2)
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
        x_coords, u_approx, u_real = galerkin(n_elem, x_start, x_end, polydeg=2)
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
