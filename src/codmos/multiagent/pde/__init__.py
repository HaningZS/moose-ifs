"""PDE verification engine for MOOSE simulation code.

Three-layer architecture:
  Layer 1 (data):    representation, kernel_map, material_resolver, boundary_matcher
  Layer 2 (engines): reconstruction, extraction, ifs_engine
  Layer 3 (bridge):  conversion (PhysicsSpecification <-> PDERepresentation)

Public API::

    from codmos.multiagent.pde import (
        PDERepresentation, PDETerm, BoundaryCondition, InitialCondition,
        reconstruct_pde, compute_ifs, IFSResult,
        physicsspec_to_pde, pde_to_physicsspec,
    )
"""

from codmos.multiagent.pde.boundary_matcher import match_boundaries
from codmos.multiagent.pde.conversion import pde_to_physicsspec, physicsspec_to_pde
from codmos.multiagent.pde.ifs_engine import (
    SEVERITY_WEIGHTS,
    GroundTruthAnnotation,
    IFSCheckpoint,
    IFSResult,
    compute_ifs,
    format_violations_for_code,
)
from codmos.multiagent.pde.kernel_map import KernelMap
from codmos.multiagent.pde.reconstruction import reconstruct_pde
from codmos.multiagent.pde.representation import (
    BoundaryCondition,
    InitialCondition,
    PDERepresentation,
    PDETerm,
)

__all__ = [
    "BoundaryCondition",
    "GroundTruthAnnotation",
    "IFSCheckpoint",
    "IFSResult",
    "InitialCondition",
    "KernelMap",
    "PDERepresentation",
    "PDETerm",
    "SEVERITY_WEIGHTS",
    "compute_ifs",
    "format_violations_for_code",
    "match_boundaries",
    "pde_to_physicsspec",
    "physicsspec_to_pde",
    "reconstruct_pde",
]
