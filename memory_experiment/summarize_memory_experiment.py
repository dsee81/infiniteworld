from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


RESULT_CSV = Path("/mnt/shared_storage/dsee/memory_experiment/results/batch_results.csv")


def _avg(rows, key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in {"", None}]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    if not RESULT_CSV.exists():
        raise SystemExit(f"Missing results CSV: {RESULT_CSV}")

    rows = list(csv.DictReader(RESULT_CSV.open()))
    ok_rows = [r for r in rows if r["status"] == "ok"]
    print(f"successful_runs={len(ok_rows)} total_rows={len(rows)}")

    by_exp = defaultdict(list)
    by_family = defaultdict(list)
    by_chunk_count = defaultdict(list)
    for row in ok_rows:
        by_exp[row["experiment"]].append(row)
        by_family[row["family"]].append(row)
        by_chunk_count[row["memory_chunks"]].append(row)

    print("\nPER_EXPERIMENT")
    for exp in sorted(by_exp):
        rs = by_exp[exp]
        print(
            exp,
            f"n={len(rs)}",
            f"comp={_avg(rs, 'composite_score'):.4f}",
            f"traj={_avg(rs, 'traj_score'):.4f}",
            f"sem={_avg(rs, 'semantic_score'):.4f}",
            f"flow={_avg(rs, 'optical_flow_score'):.4f}",
            f"mins={_avg(rs, 'runtime_minutes'):.3f}",
        )

    print("\nPER_FAMILY")
    for family in sorted(by_family):
        rs = by_family[family]
        print(
            family,
            f"n={len(rs)}",
            f"comp={_avg(rs, 'composite_score'):.4f}",
            f"traj={_avg(rs, 'traj_score'):.4f}",
            f"sem={_avg(rs, 'semantic_score'):.4f}",
            f"flow={_avg(rs, 'optical_flow_score'):.4f}",
            f"mins={_avg(rs, 'runtime_minutes'):.3f}",
        )

    print("\nCHUNK_COUNT")
    for chunk_count in sorted(by_chunk_count, key=int):
        rs = by_chunk_count[chunk_count]
        print(
            f"chunks={chunk_count}",
            f"n={len(rs)}",
            f"comp={_avg(rs, 'composite_score'):.4f}",
            f"traj={_avg(rs, 'traj_score'):.4f}",
            f"sem={_avg(rs, 'semantic_score'):.4f}",
            f"flow={_avg(rs, 'optical_flow_score'):.4f}",
            f"mins={_avg(rs, 'runtime_minutes'):.3f}",
        )


if __name__ == "__main__":
    main()
