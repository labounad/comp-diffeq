from click.testing import CliRunner
from cli.fem import fem


def test_cli_fem_solve():
    runner = CliRunner()
    result = runner.invoke(fem, ['solve', '1', 'sin(7*pi*x)', '--polydeg', '2', '--n-elems', '100'])

    assert result.exit_code == 0
    assert "Solution at midpoint (x=0.5)" in result.output
    assert "u(0.5)" in result.output
