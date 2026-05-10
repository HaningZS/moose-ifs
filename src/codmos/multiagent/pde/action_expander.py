"""Static Action expansion for reconstruction.py.

Converts common MOOSE Action blocks into synthetic kernel descriptors
so that ``_extract_terms()`` can process them through the standard
KernelMap pipeline.  Only covers deterministic expansions where the
mapping from Action parameters to kernel types is unambiguous.

Supported Actions
-----------------
- ``[Physics/SolidMechanics/QuasiStatic]``  → StressDivergenceTensors
- ``[Physics/SolidMechanics/Dynamic]``      → DynamicStressDivergenceTensors + InertialForce
- ``[SolidMechanics]`` (old, inside [Kernels]) → StressDivergenceTensors
- ``[PorousFlowBasicTHM]``                  → Darcy + MassTimeDeriv + HeatAdvection
- ``[PorousFlowFullySaturated]``            → Darcy + MassTimeDeriv + HeatAdvection + ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from codmos.multiagent.validators.hit_parser import HITNode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticKernel:
    """A kernel descriptor produced by Action expansion."""

    kernel_type: str
    variable: str
    coefficient: float | str | None = None
    coupled_variable: str | None = None
    source_action: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_actions(root: HITNode) -> list[SyntheticKernel]:
    """Walk the AST for known Action blocks and return synthetic kernels.

    Called by ``reconstruct_pde()`` *after* ``_extract_terms()`` so that
    Action-generated kernels supplement explicit ones.

    If the file already has an explicit ``[Kernels]`` block with at least
    one typed kernel entry, Action expansion is **skipped** to avoid
    double-counting in mixed files (e.g. thermal+mechanical where thermal
    is explicit and mechanics is an Action).  To process such mixed files,
    the Action part should be manually unrolled.
    """
    # Skip if explicit kernels already present
    kernels_node = root.find("Kernels")
    if kernels_node is not None:
        has_explicit = any(
            c.param("type") is not None
            for c in kernels_node.children
        )
        if has_explicit:
            return []

    results: list[SyntheticKernel] = []
    results.extend(_expand_quasi_static(root))
    results.extend(_expand_dynamic(root))
    results.extend(_expand_old_solid_mechanics(root))
    results.extend(_expand_porous_flow_basic_thm(root))
    results.extend(_expand_porous_flow_fully_saturated(root))
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_displacements(node: HITNode, root: HITNode) -> list[str]:
    """Resolve displacement variable names from an Action node or GlobalParams."""
    raw = node.param("displacements")
    if raw is None:
        # Check parent nodes up to root
        gp = root.find("GlobalParams")
        if gp is not None:
            raw = gp.param("displacements")
    if raw is None:
        return []
    return raw.strip("'\"").split()


def _is_transient(root: HITNode) -> bool:
    ex = root.find("Executioner")
    if ex is None:
        return False
    return (ex.param("type") or "").lower() == "transient"


def _find_action_sub_blocks(root: HITNode, *path_segments: str) -> list[HITNode]:
    """Find all sub-blocks of a nested Action path.

    Handles three naming patterns produced by the HIT parser:

    1. Nested blocks: ``[Physics][SolidMechanics][QuasiStatic][all]``
       → navigable via ``root.find("Physics/SolidMechanics/QuasiStatic/all")``
    2. Flat path as node name: ``[Physics/SolidMechanics/QuasiStatic]``
       → stored as a single child of root with name ``"Physics/SolidMechanics/QuasiStatic"``
    3. Mixed: ``[Physics/SolidMechanics/QuasiStatic][all]``
       → flat parent with nested child

    Returns the leaf sub-blocks that carry the actual parameters.
    """
    full_path = "/".join(path_segments)

    # Strategy 1: try root.find() (works for nested blocks)
    node = root.find(full_path)
    if node is not None:
        if node.children:
            return [c for c in node.children if c.param("type") is None]
        return [node]

    # Strategy 2: look for flat-path node name in root.children
    for child in root.children:
        if child.name == full_path:
            # Found the action as a flat-name child of root
            if child.children:
                return [c for c in child.children if c.param("type") is None]
            return [child]

    # Strategy 3: try with common sub-block names
    for sub in ("all", "All"):
        combined = f"{full_path}/{sub}"
        sub_node = root.find(combined)
        if sub_node is not None:
            return [sub_node]
        # Also check flat-name parent + nested sub-block child
        for child in root.children:
            if child.name == full_path:
                for gc in child.children:
                    if gc.name == sub:
                        return [gc]

    return []


# ---------------------------------------------------------------------------
# SolidMechanics / QuasiStatic
# ---------------------------------------------------------------------------

def _expand_quasi_static(root: HITNode) -> list[SyntheticKernel]:
    sub_blocks = _find_action_sub_blocks(root, "Physics", "SolidMechanics", "QuasiStatic")
    if not sub_blocks:
        return []

    results: list[SyntheticKernel] = []
    for sb in sub_blocks:
        displacements = _get_displacements(sb, root)
        if not displacements:
            logger.warning("QuasiStatic action found but no displacements resolved")
            continue

        use_ad = (sb.param("use_automatic_differentiation") or "false").lower() == "true"
        kernel_type = "ADStressDivergenceTensors" if use_ad else "StressDivergenceTensors"

        for disp_var in displacements:
            results.append(SyntheticKernel(
                kernel_type=kernel_type,
                variable=disp_var,
                source_action="Physics/SolidMechanics/QuasiStatic",
            ))

    if results:
        logger.info("QuasiStatic action expanded to %d StressDivergenceTensors kernels", len(results))
    return results


# ---------------------------------------------------------------------------
# SolidMechanics / Dynamic
# ---------------------------------------------------------------------------

def _expand_dynamic(root: HITNode) -> list[SyntheticKernel]:
    sub_blocks = _find_action_sub_blocks(root, "Physics", "SolidMechanics", "Dynamic")
    if not sub_blocks:
        return []

    results: list[SyntheticKernel] = []
    for sb in sub_blocks:
        displacements = _get_displacements(sb, root)
        if not displacements:
            logger.warning("Dynamic action found but no displacements resolved")
            continue

        use_ad = (sb.param("use_automatic_differentiation") or "false").lower() == "true"
        sdt_type = "ADDynamicStressDivergenceTensors" if use_ad else "DynamicStressDivergenceTensors"

        for disp_var in displacements:
            results.append(SyntheticKernel(
                kernel_type=sdt_type,
                variable=disp_var,
                source_action="Physics/SolidMechanics/Dynamic",
            ))
            results.append(SyntheticKernel(
                kernel_type="InertialForce",
                variable=disp_var,
                source_action="Physics/SolidMechanics/Dynamic",
            ))

    if results:
        logger.info("Dynamic action expanded to %d kernels", len(results))
    return results


# ---------------------------------------------------------------------------
# Old-style [SolidMechanics] / [DynamicSolidMechanics] inside [Kernels]
# ---------------------------------------------------------------------------

def _expand_old_solid_mechanics(root: HITNode) -> list[SyntheticKernel]:
    kernels_node = root.find("Kernels")
    if kernels_node is None:
        return []

    results: list[SyntheticKernel] = []
    for child in kernels_node.children:
        if child.name in ("SolidMechanics", "DynamicSolidMechanics"):
            displacements = _get_displacements(child, root)
            if not displacements:
                continue

            is_dynamic = child.name == "DynamicSolidMechanics"
            sdt_type = "DynamicStressDivergenceTensors" if is_dynamic else "StressDivergenceTensors"

            for disp_var in displacements:
                results.append(SyntheticKernel(
                    kernel_type=sdt_type,
                    variable=disp_var,
                    source_action=f"[Kernels]/{child.name}",
                ))
                if is_dynamic:
                    results.append(SyntheticKernel(
                        kernel_type="InertialForce",
                        variable=disp_var,
                        source_action=f"[Kernels]/{child.name}",
                    ))

    if results:
        logger.info("Old %s action expanded to %d kernels",
                     "DynamicSolidMechanics" if any("Dynamic" in r.source_action for r in results) else "SolidMechanics",
                     len(results))
    return results


# ---------------------------------------------------------------------------
# PorousFlowBasicTHM
# ---------------------------------------------------------------------------

def _expand_porous_flow_basic_thm(root: HITNode) -> list[SyntheticKernel]:
    node = root.find("PorousFlowBasicTHM")
    if node is None:
        return []

    pp_var = node.param("porepressure") or "porepressure"
    results: list[SyntheticKernel] = [
        SyntheticKernel(
            kernel_type="PorousFlowFullySaturatedDarcyBase",
            variable=pp_var,
            source_action="PorousFlowBasicTHM",
        ),
    ]

    if _is_transient(root):
        results.append(SyntheticKernel(
            kernel_type="PorousFlowMassTimeDerivative",
            variable=pp_var,
            source_action="PorousFlowBasicTHM",
        ))

    temp_var = node.param("temperature")
    if temp_var:
        results.append(SyntheticKernel(
            kernel_type="PorousFlowFullySaturatedHeatAdvection",
            variable=temp_var,
            source_action="PorousFlowBasicTHM",
        ))
        if _is_transient(root):
            results.append(SyntheticKernel(
                kernel_type="PorousFlowEnergyTimeDerivative",
                variable=temp_var,
                source_action="PorousFlowBasicTHM",
            ))

    logger.info("PorousFlowBasicTHM expanded to %d kernels", len(results))
    return results


# ---------------------------------------------------------------------------
# PorousFlowFullySaturated
# ---------------------------------------------------------------------------

def _expand_porous_flow_fully_saturated(root: HITNode) -> list[SyntheticKernel]:
    node = root.find("PorousFlowFullySaturated")
    if node is None:
        return []

    pp_var = node.param("porepressure") or "porepressure"

    # Stabilization affects kernel choice
    stab = (node.param("stabilization") or "none").lower()
    if stab == "full":
        darcy_type = "PorousFlowFullySaturatedAdvectiveFlux"
    elif stab == "kt":
        darcy_type = "PorousFlowFluxLimitedTVDAdvection"
    else:
        darcy_type = "PorousFlowFullySaturatedDarcyFlow"

    results: list[SyntheticKernel] = [
        SyntheticKernel(kernel_type=darcy_type, variable=pp_var,
                        source_action="PorousFlowFullySaturated"),
    ]

    if _is_transient(root):
        results.append(SyntheticKernel(
            kernel_type="PorousFlowMassTimeDerivative",
            variable=pp_var,
            source_action="PorousFlowFullySaturated",
        ))

    temp_var = node.param("temperature")
    if temp_var:
        if stab == "full":
            heat_type = "PorousFlowFullySaturatedUpwindHeatAdvection"
        elif stab == "kt":
            heat_type = "PorousFlowFluxLimitedTVDAdvection"
        else:
            heat_type = "PorousFlowFullySaturatedHeatAdvection"

        results.append(SyntheticKernel(
            kernel_type=heat_type, variable=temp_var,
            source_action="PorousFlowFullySaturated",
        ))
        if _is_transient(root):
            results.append(SyntheticKernel(
                kernel_type="PorousFlowEnergyTimeDerivative",
                variable=temp_var,
                source_action="PorousFlowFullySaturated",
            ))

    # Mechanical coupling
    coupling = (node.param("coupling_type") or "").lower()
    if "mechanical" in coupling or "thm" in coupling:
        displacements = _get_displacements(node, root)
        if displacements and _is_transient(root):
            results.append(SyntheticKernel(
                kernel_type="PorousFlowMassVolumetricExpansion",
                variable=pp_var,
                source_action="PorousFlowFullySaturated",
            ))

    logger.info("PorousFlowFullySaturated expanded to %d kernels (stabilization=%s)", len(results), stab)
    return results
