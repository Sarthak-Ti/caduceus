"""Collate tests/results/*.json into the markdown tables used by
claude_summaries/mamba_optimization_report.md.

    pixi run python -m tests.summarize_results tests/results/*.json
"""

import argparse
import glob
import json
import os
import sys


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        meta = payload.get("meta", {})
        for r in payload["results"]:
            r["_file"] = os.path.basename(p)
            r["_gpu"] = meta.get("env", {}).get("gpu", "?")
            rows.append(r)
    return rows


def md_table(rows, baseline_variant="baseline"):
    """One row per (name, variant, B, L), with speedup vs the matching baseline."""
    base = {}
    for r in rows:
        if r["variant"] == baseline_variant and not r["oom"] and not r["error"]:
            base[(r["name"], r["batch_size"], r["seqlen"])] = r

    hdr = ("| config | variant | B | L | fwd ms | bwd ms | step ms | tokens/s | "
           "peak alloc GB | speedup |")
    out = [hdr, "|" + "---|" * 10]
    key = lambda r: (r["name"], r["seqlen"], r["batch_size"], r["variant"])
    for r in sorted(rows, key=key):
        if r["oom"]:
            out.append(f"| {r['name']} | {r['variant']} | {r['batch_size']} | "
                       f"{r['seqlen']} | OOM | | | | | |")
            continue
        if r["error"]:
            out.append(f"| {r['name']} | {r['variant']} | {r['batch_size']} | "
                       f"{r['seqlen']} | ERROR | | | | | |")
            continue
        b = base.get((r["name"], r["batch_size"], r["seqlen"]))
        sp = f"{b['step_ms'] / r['step_ms']:.2f}x" if b and r["step_ms"] > 0 else "—"
        out.append(
            f"| {r['name']} | {r['variant']} | {r['batch_size']} | {r['seqlen']} | "
            f"{r['fwd_ms']:.1f} | {r['bwd_ms']:.1f} | {r['step_ms']:.1f} | "
            f"{r['tokens_per_s_step']:,.0f} | {r['peak_alloc_gb']:.2f} | {sp} |")
    return "\n".join(out)


def micro_table(rows, seqlen=None):
    """Per-op attribution, sorted by total time, with each op's share of the step."""
    micro = [r for r in rows if r["name"] not in
             ("caduceus_backbone", "joint_train_step", "backbone_batch_scaling",
              "caduceus_inference")]
    if seqlen:
        micro = [r for r in micro if r["seqlen"] == seqlen]
    micro = [r for r in micro if not r["oom"] and not r["error"]]
    if not micro:
        return "(no micro results)"
    layer = next((r for r in micro if r["name"].startswith("BiMambaWrapper")), None)
    out = ["| op | fwd ms | bwd ms | total ms | x16 layers (ms) | peak GB |",
           "|" + "---|" * 6]
    for r in sorted(micro, key=lambda r: -(r["fwd_ms"] + r["bwd_ms"])):
        tot = r["fwd_ms"] + r["bwd_ms"]
        per_model = f"{tot * 16:.0f}" if r is not layer or True else ""
        out.append(f"| {r['name']} | {r['fwd_ms']:.2f} | {r['bwd_ms']:.2f} | "
                   f"{tot:.2f} | {per_model} | {r['peak_alloc_gb']:.2f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["tests/results/*.json"])
    ap.add_argument("--micro-seqlen", type=int, default=None)
    args = ap.parse_args()
    paths = []
    for p in args.paths:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    if not paths:
        print("no result files found", file=sys.stderr)
        return 1
    rows = load(paths)
    print(f"### sources\n\n" + "\n".join(f"- `{p}`" for p in paths))
    print("\n### model-level\n")
    print(md_table([r for r in rows if r["name"] in
                    ("caduceus_backbone", "joint_train_step", "caduceus_inference",
                     "backbone_batch_scaling")]))
    print("\n### per-op attribution\n")
    print(micro_table(rows, args.micro_seqlen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
