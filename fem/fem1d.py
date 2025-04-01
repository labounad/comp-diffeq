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

from enum import Enum


class Fem1dParams:
    K_CONST = 7
    N_DIRICHLET = 2

    MIN_NODES_EXP = 3
    MAX_NODES_EXP = 12

    X_START = 0
    X_END = 1

    @staticmethod
    def default_diffusion_function(x):
        # half_nodes = len(x) // 2
        # other_half = len(x) - half_nodes
        # return np.concatenate((np.ones(half_nodes), 2 * np.ones(other_half)))
        return np.ones_like(x)

    @staticmethod
    def default_u_real(x):
        """
        u(x) = sin(k*pi*x)
        """
        return np.sin(Fem1dParams.K_CONST * math.pi * x)

    @staticmethod
    def default_source(x):
        """
        f(x) = (k*pi)^2 sin(k*pi*x)
        """
        return Fem1dParams.default_diffusion_function(x) * (Fem1dParams.K_CONST * math.pi) ** 2 * Fem1dParams.default_u_real(
            x)


class BoundaryTypes(Enum):
    DIRICHLET = 'dirichlet'
    NEUMANN = 'neumann'
    CAUCHY = 'cauchy'


class Fem1dInput:
    def __init__(self,
                 source_function=Fem1dParams.default_source,
                 diffusion_function=Fem1dParams.default_diffusion_function,
                 domain=(0, 1),
                 n_elems=10**4,
                 polydeg=1,
                 boundary_conditions=False,
                 boundary_type=BoundaryTypes.DIRICHLET.value):
        self.source = source_function
        self.diffusion = diffusion_function

        self.x_start, self.x_end = domain
        self.n_elems = n_elems

        nodes = n_elems * polydeg + 1
        self.n_nodes = nodes
        self.x_coords = np.linspace(self.x_start, self.x_end, nodes)

        self.elements = _elem_indices(n_elems, polydeg)

        self.degree = polydeg

        if not boundary_conditions:
            self.boundary_conditions = np.array([[0, nodes-1], [0, 0]])
        else:
            self.boundary_conditions = np.array(boundary_conditions)
            assert np.shape(self.boundary_conditions) == (2, 2)

        if boundary_type == BoundaryTypes.DIRICHLET.value:
            self.boundary_type = BoundaryTypes.DIRICHLET.value
            self.n_dirichlet = Fem1dParams.N_DIRICHLET
            self.dirichlet_nodes = self.boundary_conditions[0]
            self.dirichlet_vals = self.boundary_conditions[1]

        return


def _elem_indices(n: int, m: int) -> np.ndarray:
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


def _impose_dirichlet_boundary(
        n_dir: int,
        boundary_nodes: np.ndarray,
        boundary_vals: np.ndarray,
        stiff_mat: np.ndarray,
        rhs: np.ndarray,
        penalty=1e15,
        neumann=False  # TODO: REMOVE FLAG
) -> [np.ndarray, np.ndarray]:
    if neumann:
        return _impose_neumann_boundary(n_dir, penalty=penalty)  # DEFUNCT

    for idir in range(n_dir):
        iglob = boundary_nodes[idir]
        stiff_mat[iglob, iglob] = penalty
        rhs[iglob] = penalty * boundary_vals[idir]
    return stiff_mat, rhs


def impose_boundary():
    pass


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


def galerkin(inputs=Fem1dInput()):
    basis_coeff = local_basis(polydeg=inputs.degree)

    stiff_mat = build_stiffness_matrix(inputs.x_coords, inputs.degree, inputs.elements, basis_coeff, inputs.diffusion)
    load_vect = build_load_vector(
        x_coords=inputs.x_coords, d=inputs.degree, elems=inputs.elements, source=inputs.source, basis=basis_coeff
    )

    if inputs.boundary_type == BoundaryTypes.DIRICHLET.value:
        stiff_mat, load_vect = _impose_dirichlet_boundary(n_dir=inputs.n_dirichlet, boundary_nodes=inputs.dirichlet_nodes,
                                                          boundary_vals=inputs.dirichlet_vals, stiff_mat=stiff_mat,
                                                          rhs=load_vect)

    u_approx = np.linalg.solve(stiff_mat, load_vect)

    return inputs.x_coords, u_approx


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
    n_elems = [2 ** i for i in range(Fem1dParams.MIN_NODES_EXP, Fem1dParams.MAX_NODES_EXP + 1)]
    x_start = Fem1dParams.X_START
    x_end = Fem1dParams.X_END

    residuals = []

    u_real = Fem1dParams.default_u_real

    for n_elem in n_elems:
        fem_input = Fem1dInput(domain=(x_start, x_end), n_elems=n_elem, polydeg=1)
        x_coords, u_approx = galerkin(fem_input)
        plt.plot(x_coords, u_approx)

        residuals.append(calc_l2err(x_coords, u_approx, u_real))

    x_fine = np.linspace(x_start, x_end, 10 ** 4)
    plt.plot(x_fine, Fem1dParams.default_u_real(x_fine))
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
    n_elems = [2 ** i for i in range(Fem1dParams.MIN_NODES_EXP, Fem1dParams.MAX_NODES_EXP + 1)]
    x_start = Fem1dParams.X_START
    x_end = Fem1dParams.X_END

    residuals = []

    # Define marker cycle
    markers = cycle(['o', 's', '^', 'x', 'D', 'v', 'P', '*'])

    u_real = Fem1dParams.default_u_real

    for n_elem in n_elems:
        x_coords, u_approx = galerkin(n_elem)
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
    # main()
