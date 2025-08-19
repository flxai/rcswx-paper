#!/usr/bin/env python3
# scripts/einsearch-progress.py
# Fast-only (head+tail) parser. O(1) RAM per file.
# Usage:
#   scripts/einsearch-progress.py <logs_root1> [<logs_root2> ...] [--md]
#                                 [--sort-rows asc|desc] [--sort-cols asc|desc]
#                                 [--head-kb N] [--tail-kb N]
#                                 [--no-sort]
import re, sys, argparse, math, os
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# ── Regexes ────────────────────────────────────────────────────────────────────
RGX_ANSI   = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RGX_PAIR   = re.compile(r"\b(\d+)/(\d+)\b")
RGX_TIME   = re.compile(r"(ETA[^0-9]{0,10})?[< ]?[0-9]{1,3}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?[>]?$")
RGX_SEED1  = re.compile(r"--seed\s+(\d+)", re.I)
RGX_SEED2  = re.compile(r"\bseed[ \t=:]+(\d+)\b", re.I)
RGX_DATA   = re.compile(r"dataset[ \t=:]+([A-Za-z0-9_.+-]+)", re.I)
RGX_XSTRAT = re.compile(r"crossover[_ -]*strategy[ \t=:]+([A-Za-z0-9_.+-]+)", re.I)
RGX_XRATE  = re.compile(r"crossover[_ -]*rate[ \t=:]+([0-9.]+)\b", re.I)
RGX_MRATE  = re.compile(r"mutation[_ -]*rate[ \t=:]+([0-9.]+)\b", re.I)
RGX_GEN    = re.compile(r"generational[ \t=:]+(True|False)", re.I)

# Failure detectors (tail scan; add more patterns as needed)
FAIL_PATTERNS = [
    r"RuntimeError: Crossover failed to generate valid children\.",
]
FAIL_REGEXES = [re.compile(p, re.I) for p in FAIL_PATTERNS]

# Abbreviations
X_ABBR = {
    "recursive_constrained_smith_waterman": "RCSWX",
    "constrained_smith_waterman": "CSWX",
    "shortest_edit_path": "SEPX",
    "one_point": "1PX",
    "None": "None",
    "": "None",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def secs_to_hm(s: float) -> str:
    s = int(math.floor(s))
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}:{m:02d}"

def trim_zero(x: str) -> str:
    return x[:-2] if x.endswith(".0") else x

def choose_workers(num_files: int) -> int:
    c = os.cpu_count() or 1
    if c <= 1:
        n = 1
    elif c < 16:
        n = max(1, c - 2)
    else:
        n = 16
    return min(n, num_files)

# ── Fast path: read only head + tail (skips runtime sum) ───────────────────────
def extract_from_log_fast(p: Path, head_kb: int = 256, tail_kb: int = 1024):
    def tail_has_failure(t: str) -> bool:
        for ln in t.splitlines():
            for rx in FAIL_REGEXES:
                if rx.search(ln):
                    return True
        return False

    seed = -1
    dataset = ""
    xstrat = ""
    xrate = ""
    mrate = ""
    mode = "Steady-State"

    with p.open("rb") as f:
        head = f.read(head_kb * 1024).decode("utf-8", "ignore")
        m = RGX_SEED1.search(head) or RGX_SEED2.search(head);     seed    = int(m.group(1)) if m else -1
        m = RGX_DATA.search(head);                                 dataset = m.group(1) if m else ""
        m = RGX_XSTRAT.search(head);                               xstrat  = m.group(1) if m else ""
        m = RGX_XRATE.search(head);                                xrate   = m.group(1) if m else ""
        m = RGX_MRATE.search(head);                                mrate   = m.group(1) if m else ""
        m = RGX_GEN.search(head);                                  mode    = "Generational" if (m and m.group(1).lower()=="true") else "Steady-State"

        sz = f.seek(0, os.SEEK_END)
        start = max(0, sz - tail_kb * 1024)
        f.seek(start)
        tail = f.read().decode("utf-8", "ignore")

    # Find last progress in tail; prefer line that has both pair and time.
    last_val = None
    failed = tail_has_failure(tail)
    for ln in reversed(tail.replace("\r","\n").split("\n")):
        if not ln:
            continue
        if RGX_PAIR.search(ln) and RGX_TIME.search(ln):
            try:
                last_val = int(RGX_PAIR.search(ln).group(1)); break
            except Exception:
                pass
        elif last_val is None:
            m = RGX_PAIR.search(ln)
            if m:
                try: last_val = int(m.group(1))
                except Exception: pass

    if xrate in {"0","0.0"}: xstrat = "None"
    xabbr = X_ABBR.get(xstrat, xstrat)
    xrate_disp = trim_zero(xrate or "0")
    mut_disp   = trim_zero(mrate or "0")
    row_raw = (mode, xabbr, xrate_disp, mut_disp)

    return {
        "seed": seed, "dataset": dataset, "value": last_val, "runtime": 0.0, "failed": int(bool(failed)),
        "mode": mode, "xabbr": xabbr, "xrate_disp": xrate_disp, "mut_disp": mut_disp, "row_key": row_raw
    }

# ── Row/cell formatting ────────────────────────────────────────────────────────
def format_row_label(mode: str, xabbr: str, xrate: str, mut: str) -> str:
    return f"{mode:<12}  {xabbr:>5}(p={xrate})  mut={mut}"

def fmt_console_cell(val, rt, failed=False):
    # Always show value; append runtime line only if rt>0.
    if val is None or (isinstance(val, float) and pd.isna(val)):
        l1 = "[bold red]✗[/]" if failed else "[grey50]⧖[/]"
    else:
        try:
            n = int(val)
            if failed:
                l1 = f"[bold red]{n}[/]"
            else:
                l1 = f"[bold green]{n}[/]" if n >= 1000 else str(n)
        except Exception:
            l1 = f"[bold red]{val}[/]" if failed else str(val)
    l2 = f"\n[grey50]⧗ {secs_to_hm(rt)}[/]" if (rt and rt > 0) else ""
    return f"{l1}{l2}"

def fmt_md_cell(val, rt, failed=False):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        l1 = "<span style='color:red'>✗</span>" if failed else "⧖"
    else:
        try:
            n = int(val)
            base = f"**{n}**" if n >= 1000 else str(n)
            l1 = f"<span style='color:red'>{base}</span>" if failed else base
        except Exception:
            l1 = f"<span style='color:red'>{val}</span>" if failed else str(val)
    l2 = f"<br>⧗ {secs_to_hm(rt)}" if (rt and rt > 0) else ""
    return f"{l1}{l2}"

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="Directories/files with *.txt logs (searched recursively)")
    ap.add_argument("--md", action="store_true", help="Print Markdown tables")
    ap.add_argument("--sort-rows", choices=["asc", "desc"], default="asc")
    ap.add_argument("--sort-cols", choices=["asc", "desc"], default="asc")
    ap.add_argument("--head-kb", type=int, default=256, help="Head size (KiB)")
    ap.add_argument("--tail-kb", type=int, default=1024, help="Tail size (KiB)")
    ap.add_argument("--no-sort", action="store_true", help="Do not sort file list (faster on many files)")
    args = ap.parse_args()

    # Collect files from all roots (dirs or individual .txt files), dedup while preserving order.
    files = []
    seen = set()
    def add_file(p: Path):
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen and p.is_file():
            seen.add(key)
            files.append(Path(key))

    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            print(f"Not found: {root}", file=sys.stderr)
            continue
        if root.is_dir():
            for p in root.rglob("*.txt"):
                add_file(p)
        elif root.is_file() and root.suffix.lower() == ".txt":
            add_file(root)
        else:
            print(f"Skipping non-text file: {root}", file=sys.stderr)

    if not files:
        print("No files.", file=sys.stderr); sys.exit(1)
    if not args.no_sort:
        files.sort()

    parser = (lambda p: extract_from_log_fast(p, args.head_kb, args.tail_kb))

    recs = []
    n_workers = choose_workers(len(files))
    with ThreadPoolExecutor(max_workers=n_workers) as ex, tqdm(total=len(files), desc="Parsing", unit="file") as pbar:
        futs = {ex.submit(parser, p): p for p in files}
        for fu in as_completed(futs):
            try:
                r = fu.result()
                # Keep records that have a value OR clearly failed.
                if r["dataset"] and (r["value"] is not None or r.get("failed")):
                    recs.append(r)
            except Exception:
                pass
            finally:
                pbar.update(1)

    if not recs:
        print("No usable records.", file=sys.stderr); sys.exit(1)

    df = pd.DataFrame(recs)
    seeds = sorted(df["seed"].unique())

    if args.md:
        for s in seeds:
            sub = df[df.seed == s]
            pt_val = pd.pivot_table(sub, index="row_key", columns="dataset", values="value",   aggfunc="max")
            pt_rt  = pd.pivot_table(sub, index="row_key", columns="dataset", values="runtime", aggfunc="sum")
            pt_fail= pd.pivot_table(sub, index="row_key", columns="dataset", values="failed",  aggfunc="max")
            pt_val = pt_val.sort_index(ascending=(args.sort_rows == "asc"))
            cols   = sorted(pt_val.columns, reverse=(args.sort_cols == "desc"))
            pt_val = pt_val.reindex(cols, axis=1)
            pt_rt  = pt_rt.reindex(index=pt_val.index, columns=pt_val.columns)
            pt_fail= pt_fail.reindex(index=pt_val.index, columns=pt_val.columns)

            meta = sub[["row_key", "mode", "xabbr", "xrate_disp", "mut_disp"]].drop_duplicates("row_key")
            rename_map = {
                k: format_row_label(m, x, xr, mu)
                for k, m, x, xr, mu in meta[["row_key", "mode", "xabbr", "xrate_disp", "mut_disp"]].itertuples(index=False, name=None)
            }
            pt_val.rename(index=rename_map, inplace=True)
            pt_rt = pt_rt.rename(index=rename_map)
            pt_fail = pt_fail.rename(index=rename_map)

            if pt_val.empty:
                continue

            out = pt_val.astype(object)
            for r in out.index:
                for c in out.columns:
                    v = pt_val.at[r, c] if c in pt_val.columns else None
                    rt = float(pt_rt.at[r, c]) if (c in pt_rt.columns and pd.notna(pt_rt.at[r, c])) else 0.0
                    fl = bool(pt_fail.at[r, c]) if (c in pt_fail.columns and pd.notna(pt_fail.at[r, c])) else False
                    out.at[r, c] = fmt_md_cell(v, rt, fl)

            colalign = ["left"] + ["right"] * out.shape[1]
            print(f"\n### Seed {s}\n")
            print(out.to_markdown(colalign=colalign))
    else:
        console = Console()
        for s in seeds:
            sub = df[df.seed == s]
            pt_val = pd.pivot_table(sub, index="row_key", columns="dataset", values="value",   aggfunc="max")
            pt_rt  = pd.pivot_table(sub, index="row_key", columns="dataset", values="runtime", aggfunc="sum")
            pt_fail= pd.pivot_table(sub, index="row_key", columns="dataset", values="failed",  aggfunc="max")
            pt_val = pt_val.sort_index(ascending=(args.sort_rows == "asc"))
            cols   = sorted(pt_val.columns, reverse=(args.sort_cols == "desc"))
            pt_val = pt_val.reindex(cols, axis=1)
            pt_rt  = pt_rt.reindex(index=pt_val.index, columns=pt_val.columns)
            pt_fail= pt_fail.reindex(index=pt_val.index, columns=pt_val.columns)

            if pt_val.empty:
                continue

            console.rule(f"Seed {s}", align="left")
            table = Table(show_header=True, header_style="bold", show_lines=False, pad_edge=False)
            table.add_column("row", justify="left", no_wrap=True)
            for col in pt_val.columns:
                table.add_column(str(col), justify="right", no_wrap=True)

            for idx in pt_val.index:
                mode, xabbr, xrate, mut = idx
                row_label = format_row_label(mode, xabbr, xrate, mut)
                cells = []
                for col in pt_val.columns:
                    v = pt_val.at[idx, col] if col in pt_val.columns else None
                    rt = float(pt_rt.at[idx, col]) if (col in pt_rt.columns and pd.notna(pt_rt.at[idx, col])) else 0.0
                    fl = bool(pt_fail.at[idx, col]) if (col in pt_fail.columns and pd.notna(pt_fail.at[idx, col])) else False
                    cells.append(fmt_console_cell(v, rt, fl))
                table.add_row(row_label, *cells)

            console.print(table)

if __name__ == "__main__":
    main()

