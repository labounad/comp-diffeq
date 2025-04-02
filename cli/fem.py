# fem: main CLI entry-point

import click

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from cli.commands import solve


@click.group()
def fem():
    """FEM CLI: Solve differential equations numerically."""
    pass


fem.add_command(solve)

if __name__ == "__main__":
    fem()




