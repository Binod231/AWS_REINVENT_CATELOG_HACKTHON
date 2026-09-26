"""Task 1 smoke tests: the package imports and the CLI reports its version."""

from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

import replan
from replan.cli.app import app

runner = CliRunner()


def test_package_exposes_version() -> None:
    assert replan.__version__
    # Three-part semantic version, so `replan --version` output is predictable.
    assert len(replan.__version__.split(".")) == 3


def test_cli_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert replan.__version__ in result.output


def test_cli_help_lists_the_tool_name() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "replan" in result.output.lower()


def test_console_script_is_installed() -> None:
    """`replan --version` must work as an installed entry point, not just in-process."""
    proc = subprocess.run(
        [sys.executable, "-m", "replan", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert replan.__version__ in proc.stdout
