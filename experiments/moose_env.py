"""MOOSE environment detection and execution.

Supports local conda environment or explicit path.
Designed to run on any machine with a MOOSE installation.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MOOSEEnvironment:
    moose_opt_path: Path
    conda_env_name: str | None
    detected_method: str  # "explicit" | "conda" | "path" | "not_found"


def detect_moose(
    explicit_path: str | None = None,
    conda_env: str = "moose",
) -> MOOSEEnvironment:
    """Detect MOOSE environment.

    Priority: explicit path > conda env > system PATH > not_found.
    """
    # 1. Explicit path
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return MOOSEEnvironment(p, None, "explicit")

    # 2. Conda environment
    try:
        result = subprocess.run(
            ["conda", "run", "-n", conda_env, "which", "moose-opt"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            path = Path(result.stdout.strip())
            return MOOSEEnvironment(path, conda_env, "conda")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. System PATH
    moose_opt = shutil.which("moose-opt")
    if moose_opt:
        return MOOSEEnvironment(Path(moose_opt), None, "path")

    # 4. Not found
    return MOOSEEnvironment(Path(""), None, "not_found")


@dataclass
class MOOSEResult:
    success: bool | None
    stdout: str
    stderr: str
    skipped: bool


def run_moose(
    env: MOOSEEnvironment,
    input_file: Path,
    timeout: int = 120,
) -> MOOSEResult:
    """Execute MOOSE --check-input on a .i file."""
    if env.detected_method == "not_found":
        return MOOSEResult(success=None, stdout="", stderr="MOOSE not found", skipped=True)

    cmd: list[str] = []
    if env.conda_env_name:
        cmd = ["conda", "run", "-n", env.conda_env_name, str(env.moose_opt_path)]
    else:
        cmd = [str(env.moose_opt_path)]
    cmd += ["-i", str(input_file), "--check-input"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return MOOSEResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            skipped=False,
        )
    except subprocess.TimeoutExpired:
        return MOOSEResult(success=False, stdout="", stderr="timeout", skipped=False)
