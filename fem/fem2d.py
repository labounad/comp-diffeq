"""
Finite element code for solving the
2D Galerkin DiffEq (D):
    -div(Diffusion grad(u)) = f  in U ⊆ ℝ^n
    u(x) = 0  (Dirichlet BC)     in ∂U
"""

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


