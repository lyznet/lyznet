"""Smoke tests for runnable minimal workflows."""

from pathlib import Path
import subprocess
import sys


def test_system_definition_example_runs():
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            repository
            / "examples/minimal_workflows/system_definition/vdp.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ContinuousTimeSystem" in completed.stdout
    assert "ControlAffineSystem" in completed.stdout
    assert "f([1, 0]) = [ 0. -1.]" in completed.stdout
