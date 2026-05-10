"""Tests for the read-only MCS blind-spot analyzer."""

import json

from scripts import analyze_mcs_blindspots


def test_analyzer_flags_high_ifs_low_mcs_bc_coefficient_case(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "thermal_case.i"
    source.write_text(
        """
[BCs]
  [convective]
    type = ConvectiveHeatFluxBC
    variable = T
    boundary = right
    T_infinity = 300
    heat_transfer_coefficient = 50
  []
[]
""",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.i"
    candidate.write_text(
        """
[BCs]
  [convective]
    type = ConvectiveHeatFluxBC
    variable = T
    boundary = right
    T_infinity = 300
    heat_transfer_coefficient = 5
  []
[]
""",
        encoding="utf-8",
    )
    records = [{
        "id": "thermal_case",
        "method": "D",
        "llm": "deepseek-flash",
        "family": "heat_transfer",
        "complexity": "medium",
        "parse": True,
        "ifs": 0.95,
        "code_path": str(candidate),
    }]

    rows = analyze_mcs_blindspots.build_rows(
        records,
        source_dir=source_dir,
        repo_root=tmp_path,
        high_ifs=0.9,
        low_mcs=0.75,
    )
    summary = analyze_mcs_blindspots.build_summary(rows)

    assert rows[0]["high_ifs_low_mcs"] is True
    assert rows[0]["mcs"] == 0.5
    assert rows[0]["mismatch_properties"] == "bc:neumann:t:right:heat_transfer_coefficient"
    assert summary["high_ifs_low_mcs_count"] == 1
    assert summary["mismatch_buckets"] == {"bc": 1}


def test_load_records_skips_malformed_lines(tmp_path):
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text(
        json.dumps({"id": "ok"}) + "\n{bad json}\n\n",
        encoding="utf-8",
    )

    records = analyze_mcs_blindspots.load_records(jsonl)

    assert records == [{"id": "ok"}]
