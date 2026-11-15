"""Smoke tests for runnable minimal workflows."""

from contextlib import contextmanager
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import sympy as sp

import lyznet as lz


def test_quadratic_verification_example_runs():
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            repository
            / "examples/minimal_workflows/quadratic_lyapunov/vdp_lyapunov_roa_z3.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Verified level: 2.3044" in completed.stdout
    assert "Verification status: verified" in completed.stdout


def test_dreal_quadratic_example_runs():
    pytest.importorskip("dreal")
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            repository
            / "examples/minimal_workflows/quadratic_lyapunov/pendulum_lyapunov_roa_dreal.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Verified local level:" in completed.stdout
    assert "Verified outer level:" in completed.stdout
    assert "Formal claim: lyapunov" in completed.stdout


@contextmanager
def _example_import_path(directory):
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path.remove(str(directory))


def _load_example(relative_path):
    repository = Path(__file__).resolve().parents[1]
    path = repository / relative_path
    with _example_import_path(path.parent):
        return runpy.run_path(path, run_name="example_test")


def test_automatica_example_definitions_and_thresholds():
    examples = (
        (
            "examples/reproductions/automatica_2025/vdp_mu_1.py",
            ((-2.5, 2.5), (-3.5, 3.5)),
            13.1032,
        ),
        (
            "examples/reproductions/automatica_2025/vdp_mu_3.py",
            ((-3.0, 3.0), (-6.0, 6.0)),
            17.5327,
        ),
        (
            "examples/reproductions/automatica_2025/two_machine.py",
            ((-2.0, 3.0), (-3.0, 1.5)),
            8.2025,
        ),
    )

    for path, bounds, minimum_area in examples:
        namespace = _load_example(path)
        assert namespace["system"].domain.bounds == bounds
        assert namespace["MINIMUM_VERIFIED_AREA"] == minimum_area


def test_neural_example_definitions():
    examples = (
        (
            "examples/minimal_workflows/pinn_lyapunov/vdp_mu_1.py",
            ((-2.5, 2.5), (-3.5, 3.5)),
        ),
        (
            "examples/minimal_workflows/pinn_lyapunov/vdp_mu_3.py",
            ((-3.0, 3.0), (-6.0, 6.0)),
        ),
        (
            "examples/minimal_workflows/pinn_lyapunov/polynomial_2d.py",
            ((-6.0, 6.0), (-6.0, 6.0)),
        ),
        (
            "examples/minimal_workflows/pinn_lyapunov/two_machine.py",
            ((-2.0, 3.0), (-3.0, 1.5)),
        ),
        (
            "examples/minimal_workflows/pinn_lyapunov/exact_basin.py",
            ((-3.0, 3.0), (-3.0, 3.0)),
        ),
    )

    for path, bounds in examples:
        namespace = _load_example(path)
        assert namespace["system"].domain.bounds == bounds


def test_default_results_directory_is_beside_executed_script(
    monkeypatch, tmp_path
):
    import __main__
    import lyznet.workflows as workflows

    script = tmp_path / "example.py"
    monkeypatch.setattr(__main__, "__file__", str(script))

    assert workflows._default_results_directory() == tmp_path / "results"


def test_published_homogeneous_example_definitions():
    degree_five = _load_example(
        "examples/minimal_workflows/homogeneous_lyapunov/degree_five.py"
    )["system"]
    irrational = _load_example(
        "examples/minimal_workflows/homogeneous_lyapunov/irrational.py"
    )["system"]

    assert degree_five.domain.bounds == ((-100.0, 100.0),) * 2
    assert lz.homogeneous_degree(degree_five) == 5
    assert irrational.domain.bounds == ((-1.0, 1.0),) * 2
    assert lz.homogeneous_degree(irrational) == 10
    assert all(
        isinstance(expression, sp.Piecewise)
        for expression in irrational.symbolic.dynamics
    )
