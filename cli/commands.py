# CLI commands

import click
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from fem.fem1d import Fem1dInput, galerkin
from cli.utils import parse_function_string


@click.command()
@click.argument('n_dim', type=int)
@click.argument('source', type=str)
@click.option('--diffusion', default="1", help="Diffusion coefficient as a function of x (default 1).")
@click.option('--polydeg', default=2, help="Polynomial degree (default quadratic).")
@click.option('--n-elems', default=1000, help="Number of elements (default 1000).")
def solve(n_dim, source, diffusion, polydeg, n_elems):
    """Solve PDE numerically with specified dimension and source term."""
    if n_dim != 1:
        click.echo("Only 1D problems are currently supported.")
        return

    source_func = parse_function_string(source)
    diffusion_func = parse_function_string(diffusion)

    fem_input = Fem1dInput(
        domain=(0, 1),
        n_elems=n_elems,
        polydeg=polydeg,
        diffusion_function=diffusion_func,
        source_function=source_func
    )

    u_approx_func = galerkin(fem_input, return_function=True)
    midpoint = 0.5
    click.echo(f"Solution at midpoint (x={midpoint}): u({midpoint}) = {u_approx_func(midpoint):.5f}")

