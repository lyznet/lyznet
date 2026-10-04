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


def test_quadratic_verification_example_runs():
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            repository
            / "examples/minimal_workflows/quadratic_lyapunov/vdp_roa.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Verified level: 2.3044" in completed.stdout
    assert "Verification status: verified" in completed.stdout


def test_dreal_quadratic_example_runs():
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            repository
            / "examples/minimal_workflows/quadratic_lyapunov/pendulum_roa.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Verified local level:" in completed.stdout
    assert "Verified outer level:" in completed.stdout
    assert "Formal claim: lyapunov" in completed.stdout
