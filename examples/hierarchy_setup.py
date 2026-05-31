import os
import sys

import ngsolve as ng
from ngsolve import H1, InnerProduct, Mesh, grad, dx, x, y, sin, pi
from netgen.geom2d import unit_square

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from multigrid_tools_temp import (
    build_form_setup,
    build_hierarchy,
    Level,
    MultigridSolver,
    VCycleConfig,
)

# ---------------------------------------------------------------------------
# Shared problem definition
# ---------------------------------------------------------------------------
DIRICHLET = "left|right|top|bottom"
u_exact = sin(pi * x) * sin(pi * y)
rhs_cf = 2 * pi * pi * sin(pi * x) * sin(pi * y)   # = -Laplace(u_exact)

# An initial guess that mixes a smooth mode with a rough (high-frequency) mode,
# so the effect of smoothing vs. a full V-cycle is visible.
x0_cf = sin(pi * x) * sin(pi * y) + 0.3 * sin(6 * pi * x) * sin(6 * pi * y)


def poisson_bilinear(a, u, v):
    a += InnerProduct(grad(u), grad(v)) * dx


def poisson_linear(f, u, v):
    f += rhs_cf * v * dx


poisson_setup = build_form_setup(bilinear=poisson_bilinear, linear=poisson_linear)

mesh_c = Mesh(unit_square.GenerateMesh(maxh=0.05))
hierarchy = build_hierarchy(
    mesh_c,
    poisson_setup,
    n_refines=2,                 # -> 3 levels (coarse, mid, fine)
    order=1,
    dirichlet=DIRICHLET,
    dirichlet_value=0.0,
    verbose=True,
)

