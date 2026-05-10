#!/usr/bin/env python3
"""Sync expert-runnable MooseBench source files into the active 220-case data.

The input directory may contain more cases than the 220-case paper slice. This
script copies only active IDs from moosebench_clean/index.json, plus mesh/table
sidecars needed at runtime, and writes the modified active IDs for follow-up
reruns.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOOSEBENCH = ROOT / "experiments" / "moosebench"
MOOSEBENCH_CLEAN = ROOT / "experiments" / "moosebench_clean"
SOURCE_DIRS = (MOOSEBENCH / "source_files", MOOSEBENCH_CLEAN / "source_files")
MODIFIED_IDS_PATH = MOOSEBENCH_CLEAN / "modified_source_ids.json"


def _normalized_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip() + "\n"


def _active_ids() -> list[str]:
    data = json.loads((MOOSEBENCH_CLEAN / "index.json").read_text(encoding="utf-8"))
    return list(data["ids"])


def _sidecars(source_final: Path) -> list[Path]:
    return sorted(
        path
        for path in source_final.iterdir()
        if path.is_file() and path.suffix.lower() in {".e", ".csv"}
    )


def sync_sources(source_final: Path, *, dry_run: bool) -> dict:
    active = _active_ids()
    missing = [case_id for case_id in active if not (source_final / f"{case_id}.i").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} active cases missing in {source_final}: {missing}")

    modified: list[str] = []
    for case_id in active:
        final_path = source_final / f"{case_id}.i"
        current_path = MOOSEBENCH / "source_files" / f"{case_id}.i"
        current_text = _normalized_text(current_path) if current_path.exists() else None
        final_text = _normalized_text(final_path)
        if current_text != final_text:
            modified.append(case_id)
            if not dry_run:
                for dst_dir in SOURCE_DIRS:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(final_path, dst_dir / final_path.name)

    sidecars = _sidecars(source_final)
    if not dry_run:
        for dst_dir in SOURCE_DIRS:
            dst_dir.mkdir(parents=True, exist_ok=True)
            for path in sidecars:
                shutil.copy2(path, dst_dir / path.name)
        MODIFIED_IDS_PATH.write_text(json.dumps(modified, indent=2) + "\n", encoding="utf-8")

    extra = sorted(path.stem for path in source_final.glob("*.i") if path.stem not in set(active))
    return {
        "active_count": len(active),
        "modified_count": len(modified),
        "modified_ids": modified,
        "extra_nonactive_count": len(extra),
        "extra_nonactive_ids": extra,
        "sidecars": [path.name for path in sidecars],
        "modified_ids_path": str(MODIFIED_IDS_PATH),
        "dry_run": dry_run,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-final",
        type=Path,
        default=ROOT / "source_files_final",
        help="Directory containing expert-runnable final source files.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without copying files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = sync_sources(args.source_final, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
