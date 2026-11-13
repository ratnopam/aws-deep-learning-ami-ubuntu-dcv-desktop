#!/usr/bin/env python3
"""Fold `vllm bench serve` JSON results into an SLO table and plots.

Produces:
  slo_table.csv   one row per (input, output, concurrency) cell
  slo_knee.md     the derived SLO recommendation, with the knee analysis
  slo_curves.png  p99 TTFT and p99 ITL against output throughput

The knee is where latency starts rising faster than throughput improves.
Setting an SLO past it buys throughput you cannot actually serve.

UNVERIFIED: not yet run against real result files. Field names follow
`vllm bench serve` output (median_ttft_ms, p99_ttft_ms, output_throughput,
...); if a key is missing it is reported as NaN rather than crashing, so
reconcile the column list against the first real run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

FNAME_RE = re.compile(r"vllmbench_in(\d+)_out(\d+)_c(\d+)\.json$")

# (csv column, candidate keys in the vllm bench serve JSON)
FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("duration_s",          ("duration",)),
    ("completed",           ("completed",)),
    ("req_throughput",      ("request_throughput",)),
    ("out_throughput",      ("output_throughput",)),
    ("total_throughput",    ("total_token_throughput",)),
    ("ttft_median_ms",      ("median_ttft_ms",)),
    ("ttft_p90_ms",         ("p90_ttft_ms",)),
    ("ttft_p99_ms",         ("p99_ttft_ms",)),
    ("itl_median_ms",       ("median_itl_ms",)),
    ("itl_p99_ms",          ("p99_itl_ms",)),
    ("tpot_median_ms",      ("median_tpot_ms",)),
    ("tpot_p99_ms",         ("p99_tpot_ms",)),
    ("e2el_median_ms",      ("median_e2el_ms",)),
    ("e2el_p99_ms",         ("p99_e2el_ms",)),
]


def first(d: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return math.nan


def load(results_dir: Path) -> list[dict]:
    rows = []
    files = sorted(results_dir.glob("vllmbench_*.json"))
    if not files:
        sys.exit(f"No vllmbench_*.json found in {results_dir}")
    for path in files:
        m = FNAME_RE.search(path.name)
        if not m:
            print(f"skip (unparseable name): {path.name}", file=sys.stderr)
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip (unreadable): {path.name}: {exc}", file=sys.stderr)
            continue
        row = {
            "input_len": int(m.group(1)),
            "output_len": int(m.group(2)),
            "concurrency": int(m.group(3)),
        }
        for col, keys in FIELDS:
            row[col] = first(data, keys)
        rows.append(row)
    return sorted(rows, key=lambda r: (r["input_len"], r["output_len"], r["concurrency"]))


def find_knee(shape_rows: list[dict]) -> dict | None:
    """Knee = last point whose marginal throughput gain still outpaces the
    marginal p99 TTFT cost, measured as relative changes."""
    usable = [r for r in shape_rows
              if not math.isnan(r["out_throughput"]) and not math.isnan(r["ttft_p99_ms"])]
    if len(usable) < 2:
        return None
    knee = usable[0]
    for prev, cur in zip(usable, usable[1:]):
        if prev["out_throughput"] <= 0 or prev["ttft_p99_ms"] <= 0:
            continue
        tput_gain = (cur["out_throughput"] - prev["out_throughput"]) / prev["out_throughput"]
        ttft_cost = (cur["ttft_p99_ms"] - prev["ttft_p99_ms"]) / prev["ttft_p99_ms"]
        if tput_gain <= 0 or ttft_cost > tput_gain:
            break
        knee = cur
    return knee


def check_slo(rows: list[dict], targets: dict[str, float]) -> list[dict]:
    """Inverse of knee-finding: given SLO TARGETS, find the most load each
    token shape can carry while still meeting every target.

    Knee analysis answers "what SLO should we promise?". This answers the
    operational question "given a promised SLO, how much traffic fits?" --
    which is what capacity planning and EKS replica counts actually need.

    `targets` maps a csv column (e.g. 'ttft_p99_ms') to its ceiling.
    """
    out = []
    shapes = sorted({(r["input_len"], r["output_len"]) for r in rows})
    for in_len, out_len in shapes:
        srows = sorted((r for r in rows
                        if r["input_len"] == in_len and r["output_len"] == out_len),
                       key=lambda r: r["concurrency"])
        passing, first_fail, binding = [], None, None
        for r in srows:
            violations = [
                f"{col}={r[col]:.1f} > {lim:.1f}"
                for col, lim in targets.items()
                if col in r and not math.isnan(r[col]) and r[col] > lim
            ]
            if violations:
                if first_fail is None:
                    first_fail, binding = r, violations
                # Keep scanning: latency is not always monotonic in concurrency,
                # and reporting the true max feasible point matters more than
                # stopping at the first blip.
            else:
                passing.append(r)
        best = max(passing, key=lambda r: r["out_throughput"]) if passing else None
        out.append({
            "input_len": in_len, "output_len": out_len,
            "best": best, "first_fail": first_fail, "binding": binding,
            "n_passing": len(passing), "n_total": len(srows),
        })
    return out


def write_slo_report(verdicts: list[dict], targets: dict[str, float],
                     path: Path, replicas: int) -> None:
    lines = [
        "# SLO feasibility report",
        "",
        "Targets:",
        *[f"- `{c}` <= {v:g}" for c, v in sorted(targets.items())],
        "",
        f"Replicas measured: {replicas}",
        "",
        "| in | out | max concurrency | out tok/s | p99 TTFT | p99 ITL | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        b = v["best"]
        if b:
            lines.append(
                f"| {v['input_len']} | {v['output_len']} | **{b['concurrency']}** "
                f"| {b['out_throughput']:.1f} | {b['ttft_p99_ms']:.1f} | {b['itl_p99_ms']:.1f} "
                f"| MEETS SLO ({v['n_passing']}/{v['n_total']} points) |"
            )
        else:
            lines.append(
                f"| {v['input_len']} | {v['output_len']} | — | — | — | — "
                f"| **INFEASIBLE** at every tested concurrency |"
            )
    lines.append("")
    for v in verdicts:
        shape = f"{v['input_len']}/{v['output_len']}"
        if v["best"] is None:
            lines += [
                f"**{shape}: infeasible.** Even concurrency 1 misses the target"
                f"{' (' + '; '.join(v['binding']) + ')' if v['binding'] else ''}. "
                "The SLO is below this configuration's latency floor — relax the target, "
                "shorten the prompt, or change TP/dtype. More replicas will NOT help: "
                "they add throughput, not single-request speed.",
                "",
            ]
        elif v["first_fail"] is not None:
            lines += [
                f"**{shape}: binds at concurrency {v['first_fail']['concurrency']}** — "
                + "; ".join(v["binding"])
                + f". Max safe load is concurrency {v['best']['concurrency']} at "
                f"{v['best']['out_throughput']:.1f} out tok/s per replica. "
                "Scale with replicas beyond that, not concurrency.",
                "",
            ]
        else:
            lines += [
                f"**{shape}: met at every tested concurrency** (up to "
                f"{v['best']['concurrency']}). The real ceiling is higher than the "
                "sweep — extend `CONCURRENCIES` to find where it actually binds.",
                "",
            ]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def write_csv(rows: list[dict], path: Path) -> None:
    import csv
    cols = ["input_len", "output_len", "concurrency"] + [c for c, _ in FIELDS]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def write_slo(rows: list[dict], path: Path, replicas: int, hourly_cost: float | None) -> None:
    shapes = sorted({(r["input_len"], r["output_len"]) for r in rows})
    lines = [
        "# Derived SLO recommendation",
        "",
        "An SLO is only reusable with its token shape, concurrency, replica count",
        "and dtype attached. All are recorded below.",
        "",
        f"- Hardware: trn2.48xlarge, BF16, LNC2, TP8",
        f"- Replicas measured: {replicas}",
        "",
    ]
    for in_len, out_len in shapes:
        srows = [r for r in rows if r["input_len"] == in_len and r["output_len"] == out_len]
        floor = min(srows, key=lambda r: r["concurrency"])
        knee = find_knee(srows)
        lines += [
            f"## {in_len} in / {out_len} out",
            "",
            "| concurrency | p99 TTFT ms | p99 ITL ms | p99 TPOT ms | out tok/s |",
            "|---|---|---|---|---|",
        ]
        for r in srows:
            lines.append(
                f"| {r['concurrency']} | {r['ttft_p99_ms']:.1f} | {r['itl_p99_ms']:.1f} "
                f"| {r['tpot_p99_ms']:.1f} | {r['out_throughput']:.1f} |"
            )
        lines += [
            "",
            f"- **Latency floor** (concurrency {floor['concurrency']}): "
            f"p99 TTFT {floor['ttft_p99_ms']:.1f} ms, p99 ITL {floor['itl_p99_ms']:.1f} ms. "
            "No SLO can beat this.",
        ]
        if knee:
            lines.append(
                f"- **Knee** at concurrency {knee['concurrency']}: "
                f"{knee['out_throughput']:.1f} out tok/s at p99 TTFT {knee['ttft_p99_ms']:.1f} ms."
            )
            lines.append(
                f"- **Proposed SLO:** p99 TTFT <= {knee['ttft_p99_ms'] * 1.1:.0f} ms and "
                f"p99 ITL <= {knee['itl_p99_ms'] * 1.1:.0f} ms at {knee['concurrency']} "
                f"concurrent requests per TP8 replica ({in_len} in / {out_len} out, BF16). "
                "The 10% headroom absorbs run-to-run variance."
            )
            if hourly_cost and knee["out_throughput"] > 0:
                per_mtok = hourly_cost / (knee["out_throughput"] * 3600 / 1e6)
                lines.append(
                    f"- **Cost at the knee:** ${per_mtok:,.2f} per million output tokens "
                    f"(at ${hourly_cost:.2f}/hr instance, {replicas} replica(s))."
                )
        else:
            lines.append("- Knee not determinable: too few usable points in the sweep.")
        lines.append("")
    lines += [
        "## Required follow-up",
        "",
        "Re-run at 2 and 4 replicas to confirm the per-replica SLO holds under",
        "instance-level contention. A single-replica number does not predict",
        "EKS behaviour, where up to 8 TP8 replicas share one trn2.48xlarge.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def plot(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots", file=sys.stderr)
        return

    shapes = sorted({(r["input_len"], r["output_len"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for in_len, out_len in shapes:
        srows = [r for r in rows if r["input_len"] == in_len and r["output_len"] == out_len]
        x = [r["out_throughput"] for r in srows]
        label = f"{in_len}/{out_len}"
        axes[0].plot(x, [r["ttft_p99_ms"] for r in srows], marker="o", label=label)
        axes[1].plot(x, [r["itl_p99_ms"] for r in srows], marker="o", label=label)
        for ax, key in ((axes[0], "ttft_p99_ms"), (axes[1], "itl_p99_ms")):
            for r in srows:
                ax.annotate(f"c{r['concurrency']}",
                            (r["out_throughput"], r[key]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[0].set(xlabel="output throughput (tok/s)", ylabel="p99 TTFT (ms)",
                title="p99 TTFT vs throughput")
    axes[1].set(xlabel="output throughput (tok/s)", ylabel="p99 ITL (ms)",
                title="p99 ITL vs throughput")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(title="in/out tokens", fontsize=8)
    fig.suptitle("Qwen3-32B on trn2.48xlarge (BF16, TP8, LNC2) -- SLO curves")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("./results"))
    ap.add_argument("--replicas", type=int, default=1,
                    help="TP8 replicas served concurrently during the sweep")
    ap.add_argument("--hourly-cost", type=float, default=None,
                    help="Instance $/hr, to derive cost per million output tokens")
    # SLO-target mode: the inverse of knee analysis. Supply the latency you
    # must promise and get back the load that fits inside it.
    ap.add_argument("--slo-ttft-p99", type=float, default=None, metavar="MS",
                    help="Target: p99 time-to-first-token ceiling in ms")
    ap.add_argument("--slo-itl-p99", type=float, default=None, metavar="MS",
                    help="Target: p99 inter-token latency ceiling in ms")
    ap.add_argument("--slo-tpot-p99", type=float, default=None, metavar="MS",
                    help="Target: p99 time-per-output-token ceiling in ms")
    ap.add_argument("--slo-e2e-p99", type=float, default=None, metavar="MS",
                    help="Target: p99 end-to-end request latency ceiling in ms")
    args = ap.parse_args()

    rows = load(args.results_dir)
    print(f"loaded {len(rows)} result cells")
    write_csv(rows, args.results_dir / "slo_table.csv")
    write_slo(rows, args.results_dir / "slo_knee.md", args.replicas, args.hourly_cost)
    plot(rows, args.results_dir / "slo_curves.png")

    targets = {k: v for k, v in {
        "ttft_p99_ms": args.slo_ttft_p99,
        "itl_p99_ms": args.slo_itl_p99,
        "tpot_p99_ms": args.slo_tpot_p99,
        "e2el_p99_ms": args.slo_e2e_p99,
    }.items() if v is not None}
    if targets:
        verdicts = check_slo(rows, targets)
        write_slo_report(verdicts, targets,
                         args.results_dir / "slo_feasibility.md", args.replicas)
        for v in verdicts:
            b = v["best"]
            shape = f"{v['input_len']}/{v['output_len']}"
            if b:
                print(f"  {shape}: max concurrency {b['concurrency']} "
                      f"at {b['out_throughput']:.1f} out tok/s")
            else:
                print(f"  {shape}: INFEASIBLE at every tested concurrency")


if __name__ == "__main__":
    main()
