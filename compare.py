#!/usr/bin/env python3
"""
compare.py — Aggregate and compare benchmark results across machines.

Usage:
  python compare.py results_*.json
  python compare.py results_mac.json results_dgx.json results_ryzen.json

Produces a formatted summary table and saves compare_results.json.
"""

import json
import sys
from pathlib import Path

# ── Formatting ─────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def col_width(values, header, padding=2):
    return max(len(str(v)) for v in [header] + list(values)) + padding

def rank_color(rank, total):
    if rank == 1:       return GREEN
    if rank == total:   return RED
    return YELLOW

def fmt(val, decimals=1):
    if val is None: return "—"
    return f"{val:.{decimals}f}"

def section(title):
    width = 70
    print(f"\n{BOLD}{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}{RESET}")

# ── Load data ──────────────────────────────────────────────────────────────────

def load_results(paths):
    machines = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  Warning: {p} not found — skipping")
            continue
        data = json.loads(path.read_text())
        machines.append(data)
        print(f"  Loaded: {path.name}  ({data['profile']['hostname']} / {data['profile']['os']})")
    return machines

# ── LLM table ─────────────────────────────────────────────────────────────────

def print_llm_table(machines):
    section("LLM Results")

    model_keys = set()
    for m in machines:
        model_keys.update(m.get("llm", {}).keys())

    if not model_keys:
        print("  No LLM results found.")
        return

    for model_key in sorted(model_keys):
        ctx_keys = set()
        for m in machines:
            ctx_keys.update(m.get("llm", {}).get(model_key, {}).keys())

        # Derive display name from first machine that has it
        display = model_key.replace("-", " ").replace("_", " ").title()

        print(f"\n  {BOLD}{display}{RESET}")

        # Header row
        col_machine = 22
        col_val     = 12

        host_headers = [m["profile"]["hostname"][:col_machine] for m in machines]
        header = f"  {'Metric':<30}" + "".join(
            f"{h:>{col_val}}" for h in host_headers
        )
        print(f"\n{header}")
        print("  " + "─" * (30 + col_val * len(machines)))

        for ctx in sorted(ctx_keys):
            tps_vals  = []
            ttft_vals = []

            for m in machines:
                entry = m.get("llm", {}).get(model_key, {}).get(ctx, {})
                tps_vals.append(entry.get("tps_mean"))
                ttft_vals.append(entry.get("ttft_mean_sec"))

            # TPS — higher is better
            print(f"  {f'TPS (gen) @ {ctx}':<30}", end="")
            valid_tps = [v for v in tps_vals if v is not None]
            best_tps  = max(valid_tps) if valid_tps else None
            for i, v in enumerate(tps_vals):
                if v is None:
                    print(f"{'—':>{col_val}}", end="")
                else:
                    color = GREEN if v == best_tps else RESET
                    print(f"{color}{fmt(v):>{col_val}}{RESET}", end="")
            print()

            # TTFT — lower is better
            print(f"  {f'TTFT (sec) @ {ctx}':<30}", end="")
            valid_ttft = [v for v in ttft_vals if v is not None]
            best_ttft  = min(valid_ttft) if valid_ttft else None
            for i, v in enumerate(ttft_vals):
                if v is None:
                    print(f"{'—':>{col_val}}", end="")
                else:
                    color = GREEN if v == best_ttft else RESET
                    print(f"{color}{fmt(v, 2):>{col_val}}{RESET}", end="")
            print()

        print()

# ── Embeddings table ───────────────────────────────────────────────────────────

def print_embedding_table(machines):
    section("Embeddings Results")

    batch_keys = set()
    for m in machines:
        batch_keys.update(m.get("embeddings", {}).keys())

    if not batch_keys:
        print("  No embedding results found.")
        return

    col_machine = 22
    col_val     = 14

    host_headers = [m["profile"]["hostname"][:col_machine] for m in machines]
    header = f"  {'Batch Size':<20}" + "".join(
        f"{h:>{col_val}}" for h in host_headers
    )
    print(f"\n{header}")
    print("  " + "─" * (20 + col_val * len(machines)))

    # Collect device tags per machine per batch key
    for bk in sorted(batch_keys, key=lambda x: int(x.split("_")[1])):
        bs = bk.split("_")[1]
        vals = []
        devices = []
        for m in machines:
            entry = m.get("embeddings", {}).get(bk, {})
            vals.append(entry.get("sentences_per_sec_mean"))
            devices.append(entry.get("device", "?"))

        # Only rank GPU results against each other — CPU results are not comparable
        gpu_vals = [v for v, d in zip(vals, devices) if v is not None and d != "cpu"]
        best = max(gpu_vals) if gpu_vals else None

        print(f"  {f'batch={bs} (sent/sec)':<20}", end="")
        for v, device in zip(vals, devices):
            if v is None:
                print(f"{'—':>{col_val}}", end="")
            elif device == "cpu":
                # Show value but dim it and mark as CPU so it's not ranked
                print(f"{DIM}{fmt(v, 0)+' (cpu)':>{col_val}}{RESET}", end="")
            else:
                color = GREEN if v == best else RESET
                print(f"{color}{fmt(v, 0):>{col_val}}{RESET}", end="")
        print()

    # Print a note if any machine used CPU
    cpu_hosts = [
        m["profile"]["hostname"]
        for m in machines
        if any(
            m.get("embeddings", {}).get(bk, {}).get("device") == "cpu"
            for bk in batch_keys
        )
    ]
    if cpu_hosts:
        print(f"  {YELLOW}Note:{RESET} {', '.join(cpu_hosts)} ran embeddings on CPU.")
        print( "  CPU results are shown dimmed and excluded from rankings.")

    print()

# ── Image table ────────────────────────────────────────────────────────────────

def print_image_table(machines):
    section("Image Generation Results  (seconds/image, lower = better)")

    # Collect all model short keys across machines
    model_keys = set()
    for m in machines:
        model_keys.update(m.get("images", {}).keys())

    if not model_keys:
        print("  No image results found (ComfyUI may not have been running).")
        return

    col_machine = 22
    col_val     = 14

    host_headers = [m["profile"]["hostname"][:col_machine] for m in machines]

    for mk in sorted(model_keys):
        # Get label from first machine that has it
        label = mk
        for m in machines:
            entry = m.get("images", {}).get(mk, {})
            if entry.get("label"):
                label = f"{entry['label']} ({entry.get('steps', '?')} steps)"
                break

        print(f"\n  {BOLD}{label}{RESET}")
        header = f"  {'Resolution':<18}" + "".join(
            f"{h:>{col_val}}" for h in host_headers
        )
        print(header)
        print("  " + "─" * (18 + col_val * len(machines)))

        # Collect all resolution keys for this model
        res_keys = set()
        for m in machines:
            res_keys.update(
                m.get("images", {}).get(mk, {}).get("resolutions", {}).keys()
            )

        for rk in sorted(res_keys):
            vals = []
            for m in machines:
                res = (m.get("images", {})
                         .get(mk, {})
                         .get("resolutions", {})
                         .get(rk, {}))
                vals.append(res.get("sec_per_image_mean"))

            print(f"  {rk:<18}", end="")
            valid = [v for v in vals if v is not None]
            best  = min(valid) if valid else None
            for v in vals:
                if v is None:
                    print(f"{'—':>{col_val}}", end="")
                else:
                    color = GREEN if v == best else RESET
                    print(f"{color}{fmt(v, 1):>{col_val}}{RESET}", end="")
            print()

    print()

# ── System summary ─────────────────────────────────────────────────────────────

def print_system_summary(machines):
    section("System Summary")
    col = 20
    fields = [
        ("hostname",  "Host"),
        ("os",        "OS"),
        ("arch",      "Arch"),
        ("backend",   "Backend"),
        ("ram_gb",    "RAM (GB)"),
        ("timestamp", "Run time"),
    ]
    for key, label in fields:
        print(f"  {label:<12}", end="")
        for m in machines:
            val = str(m["profile"].get(key, "?"))
            if key == "timestamp":
                val = val[:16].replace("T", " ")
            print(f"  {val:<{col}}", end="")
        print()
    print()

# ── Speedup summary ────────────────────────────────────────────────────────────

def print_speedup_summary(machines):
    """Show relative speedup vs the slowest machine."""
    section("Relative Speedup  (vs slowest machine, higher = faster)")

    hosts = [m["profile"]["hostname"] for m in machines]
    col   = 16

    print(f"\n  {'Metric':<40}" + "".join(f"{h[:col]:>{col}}" for h in hosts))
    print("  " + "─" * (40 + col * len(machines)))

    def speedup_row(label, values):
        """values: list of floats, higher_is_better."""
        if not any(v is not None for v in values):
            return
        valid = [v for v in values if v is not None]
        worst = min(valid)
        if worst == 0:
            return
        print(f"  {label:<40}", end="")
        for v in values:
            if v is None:
                print(f"{'—':>{col}}", end="")
            else:
                ratio = v / worst
                color = GREEN if ratio == max(v2/worst for v2 in valid) else RESET
                print(f"{color}{ratio:>{col}.2f}x{RESET}", end="")
        print()

    def speedup_row_lower(label, values):
        """values: list of floats, lower_is_better → invert for speedup."""
        if not any(v is not None for v in values):
            return
        valid = [v for v in values if v is not None]
        worst = max(valid)
        if worst == 0:
            return
        print(f"  {label:<40}", end="")
        for v in values:
            if v is None:
                print(f"{'—':>{col}}", end="")
            else:
                ratio = worst / v
                color = GREEN if ratio == max(worst/v2 for v2 in valid) else RESET
                print(f"{color}{ratio:>{col}.2f}x{RESET}", end="")
        print()

    # LLM TPS
    for model_key in set(k for m in machines for k in m.get("llm", {})):
        for ctx in set(c for m in machines for c in m.get("llm", {}).get(model_key, {})):
            vals = [m.get("llm", {}).get(model_key, {}).get(ctx, {}).get("tps_mean")
                    for m in machines]
            speedup_row(f"TPS {model_key} @ {ctx}", vals)

    # Embeddings
    for bk in set(k for m in machines for k in m.get("embeddings", {})):
        bs = bk.split("_")[1]
        vals = [m.get("embeddings", {}).get(bk, {}).get("sentences_per_sec_mean")
                for m in machines]
        speedup_row(f"Embeddings batch={bs}", vals)

    # Images (lower sec = faster) — iterate model then resolution
    for mk in set(k for m in machines for k in m.get("images", {})):
        label = mk
        for m in machines:
            if m.get("images", {}).get(mk, {}).get("label"):
                label = m["images"][mk]["label"]
                break
        res_keys = set(
            k for m in machines
            for k in m.get("images", {}).get(mk, {}).get("resolutions", {})
        )
        for rk in res_keys:
            vals = [
                m.get("images", {}).get(mk, {}).get("resolutions", {})
                 .get(rk, {}).get("sec_per_image_mean")
                for m in machines
            ]
            speedup_row_lower(f"Image {label} @ {rk}", vals)

    print()

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare.py results_*.json")
        print("       python compare.py results_mac.json results_dgx.json results_ryzen.json")
        sys.exit(1)

    paths = sys.argv[1:]

    print(f"\n{BOLD}Benchmark Comparison{RESET}")
    print("─" * 50)
    machines = load_results(paths)

    if not machines:
        print("No valid result files found.")
        sys.exit(1)

    print_system_summary(machines)
    print_llm_table(machines)
    print_embedding_table(machines)
    print_image_table(machines)
    print_speedup_summary(machines)

    # Save combined JSON
    out = {
        "machines": [m["profile"] for m in machines],
        "results":  machines,
    }
    out_path = "compare_results.json"
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"  {GREEN}✓{RESET}  Full comparison saved to {out_path}\n")

    print(f"  {DIM}Green = best  |  Red = slowest  |  — = not tested{RESET}\n")

if __name__ == "__main__":
    main()
