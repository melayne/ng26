"""Tests for heterogeneous (per-boundary / spatially varying) Dirichlet BCs.

Verifies that ``Level.enforce_dirichlet`` writes the correct values onto the
fixed DOFs of each named boundary while leaving the free (interior) DOFs alone.

Run with:  pytest tests/ -v       (from the repo root)
       or:  python tests/test_dirichlet_bcs.py
"""
import os
import sys

import numpy as np
import pytest

import ngsolve as ng
from ngsolve import H1, InnerProduct, Mesh, grad, dx, x, y
from netgen.geom2d import unit_square

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from multigrid_cycles import Level, build_form_setup  # noqa: E402

DIRICHLET = "left|right|top|bottom"
SENTINEL = 9.0  # arbitrary interior value used to detect untouched free DOFs


def _poisson_setup():
    def bilinear(a, u, v):
        a += InnerProduct(grad(u), grad(v)) * dx

    def linear(f, u, v):
        f += 1.0 * v * dx

    return build_form_setup(bilinear=bilinear, linear=linear)


def make_level(dirichlet_value, *, maxh=0.25, dirichlet=DIRICHLET) -> Level:
    """Assemble a single Poisson level on the unit square with given BC data."""
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes = H1(mesh, order=1, dirichlet=dirichlet)
    setup = _poisson_setup()
    a, f = setup(fes)
    a.Assemble()
    f.Assemble()
    return Level.from_forms(
        mesh, fes, a, f, dirichlet_value=dirichlet_value, dirichlet=dirichlet,
        built_P=False,
    )


def _exclusive_ids(level, names):
    """Map each boundary name -> its DOFs that lie on no other listed boundary.

    Corner DOFs are shared by two edges; the dict enforcement resolves them with
    a last-writer-wins rule, so we test the unambiguous (non-corner) DOFs
    strictly and the shared ones more loosely.
    """
    ids = {n: level.dirichlet_ids(n) for n in names}
    exclusive = {}
    for n in names:
        others = np.concatenate([ids[m] for m in names if m != n]) if len(names) > 1 else np.array([], int)
        exclusive[n] = np.setdiff1d(ids[n], others)
    return ids, exclusive


def test_per_boundary_constants():
    """A dict of distinct constants must land on the right boundary DOFs."""
    spec = {"left": 1.0, "right": 2.0, "top": 3.0, "bottom": 4.0}
    level = make_level(spec)

    level.gfu.vec.FV().NumPy()[:] = SENTINEL  # mark interior
    level.enforce_dirichlet()
    vals = level.gfu.vec.FV().NumPy()

    ids, exclusive = _exclusive_ids(level, list(spec))

    # Every boundary contributes at least one fixed DOF on this mesh.
    for name in spec:
        assert len(ids[name]) > 0, f"no fixed DOFs found on {name!r}"

    # Non-corner DOFs must match their boundary's value exactly.
    for name, value in spec.items():
        excl = exclusive[name]
        assert np.allclose(vals[excl], value), (
            f"{name!r} interior-edge DOFs should be {value}, got {vals[excl]}"
        )

    # Corner (shared) DOFs must equal one of their two adjacent boundary values.
    for dof in level.fixed_ids:
        allowed = {spec[n] for n in spec if dof in set(ids[n])}
        assert any(np.isclose(vals[dof], a) for a in allowed), (
            f"fixed DOF {dof} = {vals[dof]} not in adjacent values {allowed}"
        )

    # Free (interior) DOFs must be untouched.
    assert np.allclose(vals[level.free_ids], SENTINEL)

    # Every fixed DOF lies on some named boundary (full coverage, no leftovers).
    covered = np.unique(np.concatenate([ids[n] for n in spec]))
    assert np.array_equal(np.sort(level.fixed_ids), covered)


def test_interior_preserved():
    """enforce_dirichlet must only write fixed DOFs, never the free ones."""
    level = make_level({"left": 1.0, "right": 2.0, "top": 3.0, "bottom": 4.0})
    level.gfu.vec.FV().NumPy()[:] = SENTINEL
    free_before = level.gfu.vec.FV().NumPy()[level.free_ids].copy()
    level.enforce_dirichlet()
    free_after = level.gfu.vec.FV().NumPy()[level.free_ids]
    assert np.array_equal(free_before, free_after)


def test_spatially_varying_coefficient_function():
    """A CoefficientFunction BC must match a reference projection on fixed DOFs."""
    g = 1.0 + x + 2.0 * y  # smooth, globally defined -> consistent at corners
    level = make_level(g)

    level.gfu.vec.FV().NumPy()[:] = 0.0
    level.enforce_dirichlet()
    vals = level.gfu.vec.FV().NumPy()

    # Reference: interpolate g everywhere; for order-1 H1 the nodal values on the
    # boundary coincide with the boundary projection used by enforce_dirichlet.
    gref = ng.GridFunction(level.fes)
    gref.Set(g)
    ref = gref.vec.FV().NumPy()

    fixed = level.fixed_ids
    assert np.allclose(vals[fixed], ref[fixed], atol=1e-10), (
        f"max diff = {np.max(np.abs(vals[fixed] - ref[fixed]))}"
    )
    # Interior stays at the value we set (0), not touched by the boundary CF.
    assert np.allclose(vals[level.free_ids], 0.0)


def test_per_boundary_coefficient_function():
    """Dict values may themselves be CoefficientFunctions on each boundary."""
    spec = {"left": 0.0, "right": 0.0, "top": x, "bottom": 5.0 * x}
    level = make_level(spec)
    level.gfu.vec.FV().NumPy()[:] = 0.0
    level.enforce_dirichlet()
    vals = level.gfu.vec.FV().NumPy()

    # Compare the "top" edge against a reference projection of x on that edge.
    ids, exclusive = _exclusive_ids(level, list(spec))
    top_ref = ng.GridFunction(level.fes)
    top_ref.Set(x, definedon=level.mesh.Boundaries("top"))
    assert np.allclose(vals[exclusive["top"]],
                       top_ref.vec.FV().NumPy()[exclusive["top"]], atol=1e-10)


def test_unlisted_boundary_is_zeroed_with_warning():
    """Omitting a Dirichlet boundary from the dict zeros it and warns."""
    spec = {"left": 1.0, "right": 2.0, "top": 3.0}  # 'bottom' omitted
    level = make_level(spec)
    level.gfu.vec.FV().NumPy()[:] = SENTINEL

    with pytest.warns(UserWarning):
        level.enforce_dirichlet()

    vals = level.gfu.vec.FV().NumPy()
    ids, exclusive = _exclusive_ids(level, ["left", "right", "top", "bottom"])
    assert np.allclose(vals[exclusive["bottom"]], 0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
