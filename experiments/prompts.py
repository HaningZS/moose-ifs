"""Prompt templates for code generation and refinement.

Separated from variant logic so prompts can be reviewed and tuned independently.
"""

from __future__ import annotations

from codmos.multiagent.pde.representation import PDERepresentation

_CODEGEN_SYSTEM = """\
You are an expert MOOSE framework developer. Generate a complete, valid MOOSE input file (.i format) for the given physics simulation request.

Requirements:
- Output ONLY the .i file content, no explanations
- Include all required blocks: [Mesh], [Variables], [Kernels], [BCs], [Materials], [Executioner], [Outputs]
- Use correct MOOSE syntax (HIT format)
- CRITICAL — coefficients: extract every numerical parameter value (diffusivity, conductivity, viscosity, Young's modulus, Poisson's ratio, permeability, etc.) stated in the description and use it VERBATIM in [Materials]. Never substitute 1.0 or any default value when a specific value is given.
"""


def format_codegen_prompt(nl_description: str) -> str:
    """Format a natural language description into a code generation user prompt."""
    return f"""\
Generate a MOOSE input file for the following simulation:

{nl_description}

Output the complete .i file:"""


def format_spec_guided_prompt(pde_llm: PDERepresentation) -> str:
    """Format a PDE specification into a code generation user prompt.

    Used by Variant B and D — the LLM receives PDE structure, not raw NL.
    """
    lines = ["Generate a MOOSE input file that solves the following PDE system:", ""]

    # Variables
    lines.append(f"Variables: {', '.join(pde_llm.variables)}")
    lines.append(f"Domain: {pde_llm.dimensions}D")
    lines.append(f"Time scheme: {pde_llm.time_scheme}")
    lines.append("")

    # Terms
    lines.append("PDE terms:")
    for term in pde_llm.terms:
        coeff_str = f" (coefficient={term.coefficient})" if term.coefficient != 1.0 else ""
        lines.append(f"  - {term.operator} on {term.variable}{coeff_str}")
    lines.append("")

    # BCs
    if pde_llm.boundary_conditions:
        lines.append("Boundary conditions:")
        for bc in pde_llm.boundary_conditions:
            lines.append(f"  - {bc.bc_type} on '{bc.boundary}': {bc.variable} = {bc.value}")
        lines.append("")

    # ICs
    if pde_llm.initial_conditions:
        lines.append("Initial conditions:")
        for ic in pde_llm.initial_conditions:
            lines.append(f"  - {ic.variable} = {ic.value} ({ic.ic_type})")
        lines.append("")

    lines.append("Output the complete .i file:")
    return "\n".join(lines)


def format_refinement_prompt(
    nl_description: str,
    previous_code: str,
    violation_report: str,
) -> str:
    """Format a refinement prompt with PDE violation details.

    Used by Variant C and D when IFS < threshold.
    """
    return f"""\
The following MOOSE input file has physics fidelity issues. Fix them.

## Original request
{nl_description}

## Current code (with issues)
```
{previous_code}
```

## PDE Verification Report (violations found)
{violation_report}

## Instructions
Fix ONLY the identified violations. Keep correct parts unchanged.
Output the complete corrected .i file:"""


def format_generic_refinement_prompt(
    nl_description: str,
    previous_code: str,
) -> str:
    """Generic refinement prompt (ablation D-generic).

    Provides no specific PDE violation details — just "fix errors".
    """
    return f"""\
The following MOOSE input file has errors. Please fix them.

## Original request
{nl_description}

## Current code
```
{previous_code}
```

Fix the errors and output the complete corrected .i file:"""


def format_violations(ifs_result) -> str:
    """Format IFSResult checkpoints into a human-readable violation report."""
    lines = []
    for cp in ifs_result.checkpoints:
        if not cp.passed:
            lines.append(f"- [{cp.severity.upper()}] {cp.dimension}: {cp.description}")
            if cp.detail:
                lines.append(f"  Detail: {cp.detail}")
    if not lines:
        return "No violations found."
    return "\n".join(lines)
