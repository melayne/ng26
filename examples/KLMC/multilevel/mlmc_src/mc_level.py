"""Deterministic Darcy solve used by either KL- or SPDE-driven MLMC."""

from __future__ import annotations

from dataclasses import dataclass

import ngsolve as ng
import numpy as np
from netgen.geom2d import unit_square

from src.multigrid_cycles import (
    MultigridHierarchy,
    MultigridSolver,
    VCycleConfig,
    build_hierarchy,
)


def _quantity_of_interest(
    solution: ng.GridFunction,
    mesh: ng.Mesh,
    kappa: ng.CoefficientFunction,
) -> float:
    """Return the right-boundary outflow ``integral -kappa grad(p).n``."""
    normal = ng.specialcf.normal(mesh.dim)
    flux_density = -kappa * ng.InnerProduct(ng.grad(solution), normal)
    return float(
        ng.Integrate(
            ng.BoundaryFromVolumeCF(flux_density),
            mesh.Boundaries("right"),
        )
    )


@dataclass
class MCLevel:
    """A private multigrid hierarchy whose finest grid represents ``Q_l``.

    An object for level ``l`` contains levels ``0, ..., l`` because the
    existing V-cycle uses all coarser levels. It is independent of the
    ``MCLevel`` objects owned by other MLMC correction terms.
    """

    level_index: int
    hierarchy: MultigridHierarchy
    solver_config: VCycleConfig
    relative_solver_tolerance: float
    maximum_vcycles: int
    conductivity_fields: tuple[ng.GridFunction, ...]

    @classmethod
    def build(
        cls,
        *,
        coarse_maxh: float,
        level_index: int,
        relative_solver_tolerance: float,
        maximum_vcycles: int,
        pre_sweeps: int,
        post_sweeps: int,
        coarse_direct: bool,
        coarse_sweeps: int,
        dirichlet: str,
        dirichlet_value: dict[str, float],
        order: int,
    ) -> "MCLevel":
        """Build the hierarchy needed to solve one spatial level ``Q_l``."""
        if level_index < 0:
            raise ValueError("level_index must be nonnegative.")
        if not np.isfinite(coarse_maxh) or coarse_maxh <= 0.0:
            raise ValueError("coarse_maxh must be finite and positive.")
        if order < 1:
            raise ValueError("order must be positive.")
        if (
            not np.isfinite(relative_solver_tolerance)
            or relative_solver_tolerance <= 0.0
        ):
            raise ValueError(
                "relative_solver_tolerance must be finite and positive."
            )
        if maximum_vcycles < 1:
            raise ValueError("maximum_vcycles must be positive.")

        conductivity_fields: list[ng.GridFunction] = []

        def form_setup(fes):
            # Piecewise constants preserve the positivity of a positive
            # input VoxelCoefficient. An H1 interpolation can overshoot and
            # create negative conductivity values.
            coefficient_space = ng.L2(fes.mesh, order=0)
            conductivity = ng.GridFunction(coefficient_space)
            conductivity.Set(ng.CoefficientFunction(1.0))
            conductivity_fields.append(conductivity)

            u, v = fes.TnT()
            a = ng.BilinearForm(fes, symmetric=True)
            a += (
                conductivity
                * ng.InnerProduct(ng.grad(u), ng.grad(v))
                * ng.dx
            )

            # The Cliffe et al. pressure-driven example has no volume source.
            f = ng.LinearForm(fes)
            return a, f

        coarse_mesh = ng.Mesh(
            unit_square.GenerateMesh(maxh=coarse_maxh)
        )
        hierarchy = build_hierarchy(
            coarse_mesh,
            form_setup,
            n_refines=level_index,
            order=order,
            dirichlet=dirichlet,
            dirichlet_value=dirichlet_value,
            verbose=False,
        )

        if len(conductivity_fields) != hierarchy.nlevels:
            raise RuntimeError(
                "Expected one conductivity field per hierarchy level."
            )

        return cls(
            level_index=level_index,
            hierarchy=hierarchy,
            solver_config=VCycleConfig(
                pre_sweeps=pre_sweeps,
                post_sweeps=post_sweeps,
                coarse_direct=coarse_direct,
                coarse_sweeps=coarse_sweeps,
            ),
            relative_solver_tolerance=relative_solver_tolerance,
            maximum_vcycles=maximum_vcycles,
            conductivity_fields=tuple(conductivity_fields),
        )

    def update_conductivity(
        self,
        kappa: ng.CoefficientFunction,
    ) -> None:
        """Represent one physical coefficient on every required mesh.

        ``GridFunction.Set`` accepts any NGSolve ``CoefficientFunction``,
        including a ``VoxelCoefficient``. The same coordinate-based field is
        reevaluated on each mesh; it is not tied to the discretization that
        created it. Each stiffness matrix and its cached multigrid data must
        then be rebuilt.
        """
        if len(self.conductivity_fields) != self.hierarchy.nlevels:
            raise RuntimeError(
                "Expected one conductivity field per hierarchy level."
            )

        for level, conductivity in zip(
            self.hierarchy.levels,
            self.conductivity_fields,
            strict=True,
        ):
            conductivity.Set(kappa)
            values = conductivity.vec.FV().NumPy()
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise RuntimeError(
                    f"Level {self.level_index} received a nonfinite or "
                    "nonpositive conductivity representation."
                )
            level.a.Assemble()
            level.refresh()

    def reset_solutions(self) -> None:
        """Zero every stored iterate and restore its Dirichlet values."""
        for level in self.hierarchy.levels:
            level.gfu.vec.FV().NumPy()[:] = 0.0
            level.enforce_dirichlet(level.gfu.vec)

    def solve_q0_direct(self) -> ng.GridFunction:
        """Directly solve the coarsest physical problem ``Q_0``."""
        if self.level_index != 0:
            raise RuntimeError(
                "solve_q0_direct may only be called for level zero."
            )

        level = self.hierarchy.coarsest
        residual = level.residual(level.f.vec, level.gfu.vec)
        correction = level.gfu.vec.CreateVector()
        correction.data = (
            level.a.mat.Inverse(level.fes.FreeDofs()) * residual
        )
        level.gfu.vec.data += correction
        level.enforce_dirichlet(level.gfu.vec)
        return level.gfu

    def solve_with_vcycles(self) -> ng.GridFunction:
        """Solve ``Q_l``, ``l > 0``, with repeated full V-cycles."""
        if self.level_index == 0:
            raise RuntimeError("Level zero must be solved directly.")

        level = self.hierarchy.finest
        initial_residual = float(level.residual_norm())
        if not np.isfinite(initial_residual):
            raise RuntimeError("The initial residual is nonfinite.")
        if initial_residual == 0.0:
            return level.gfu

        solver = MultigridSolver(self.hierarchy, self.solver_config)
        solver.solve(
            max_cycles=self.maximum_vcycles,
            tol=self.relative_solver_tolerance,
            norms=("l2",),
            stop_norm="l2",
            verbose=False,
        )

        final_residual = float(level.residual_norm())
        relative_residual = final_residual / initial_residual
        if (
            not np.isfinite(final_residual)
            or not np.isfinite(relative_residual)
        ):
            raise RuntimeError(
                f"Level {self.level_index} produced a nonfinite residual."
            )
        if relative_residual > self.relative_solver_tolerance:
            raise RuntimeError(
                f"Level {self.level_index} did not converge after "
                f"{self.maximum_vcycles} V-cycles: relative residual="
                f"{relative_residual:.3e}, required="
                f"{self.relative_solver_tolerance:.3e}."
            )
        return level.gfu

    def solve(self) -> ng.GridFunction:
        """Reset and solve this object's finest physical problem."""
        self.reset_solutions()
        if self.level_index == 0:
            return self.solve_q0_direct()
        return self.solve_with_vcycles()

    def evaluate_qoi(self) -> float:
        """Evaluate the right-boundary outflow flux on the finest grid."""
        level = self.hierarchy.finest
        return _quantity_of_interest(
            level.gfu,
            level.mesh,
            self.conductivity_fields[-1],
        )

    def solve_qoi(
        self,
        kappa: ng.CoefficientFunction,
    ) -> float:
        """Update the coefficient, solve, and evaluate ``Q_l``."""
        self.update_conductivity(kappa)
        self.solve()
        return self.evaluate_qoi()


@dataclass(frozen=True)
class MCLevelFactory:
    """Configuration object that constructs fresh ``MCLevel`` instances."""

    coarse_maxh: float
    relative_solver_tolerance: float
    maximum_vcycles: int
    pre_sweeps: int
    post_sweeps: int
    coarse_direct: bool
    coarse_sweeps: int
    dirichlet: str
    dirichlet_value: dict[str, float]
    order: int

    def __call__(self, level_index: int) -> MCLevel:
        return MCLevel.build(
            coarse_maxh=self.coarse_maxh,
            level_index=level_index,
            relative_solver_tolerance=self.relative_solver_tolerance,
            maximum_vcycles=self.maximum_vcycles,
            pre_sweeps=self.pre_sweeps,
            post_sweeps=self.post_sweeps,
            coarse_direct=self.coarse_direct,
            coarse_sweeps=self.coarse_sweeps,
            dirichlet=self.dirichlet,
            dirichlet_value=self.dirichlet_value,
            order=self.order,
        )


__all__ = ["MCLevel", "MCLevelFactory"]
