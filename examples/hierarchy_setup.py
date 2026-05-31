#%%
import os
import sys

import ngsolve as ng
from ngsolve import H1, InnerProduct, Mesh, grad, dx, x, y, sin, pi
from ngsolve.webgui import Draw
from netgen.geom2d import unit_square

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from multigrid_cycles import (
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
    dirichlet_value={"left": 1.0, "right": 1.0, "top": 1.0, "bottom": 0.0},
    verbose=True,
)


# %%
<<<<<<< HEAD
# ---------------------------------------------------------------------------
# Set the initial guess on the finest level and draw it.
# ---------------------------------------------------------------------------
fine = hierarchy.finest
fine.set_initial_guess(x0_cf)        # interpolates x0_cf and pins Dirichlet DOFs

scene = Draw(
    fine.gfu,
    fine.mesh,
    "initial guess (finest)",
    deformation=True,
    settings={"camera": {"transformations": [{"type": "rotateX", "angle": -45}]}},
)

# %%
=======
>>>>>>> f42c5f4cdd01f536588b05e19ef55690053138f3
