#!/usr/bin/env python3
# scripts/einsearch-progress.py
# Usage: scripts/einsearch-progress.py <logs_root> [--md] [--sort-rows asc|desc] [--sort-cols asc|desc]
import re, sys, argparse, math
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

# ── Regexes ────────────────────────────────────────────────────────────────────
RGX_ANSI   = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RGX_PAIR   = re.compile(r"\b(\d+)/(\d+)\b")
RGX_TIME   = re.compile(r"(ETA[^0-9]{0,10})?[< ]?[0-9]{1,3}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?[>]?$")
RGX_SEED1  = re.compile(r"--seed\s+(\d+)", re.I)
RGX_SEED2  = re.compile(r"\bseed[ \t=:]+(\d+)\b", re.I)
RGX_DATA   = re.compile(r"dataset[ \t=:]+([A-Za-z0-9_.+-]+)", re.I)
RGX_XSTRAT = re.compile(r"crossover[_ -]*strategy[ \t=:]+([A-Za-z0-9_.+-]+)", re.I)
RGX_XRATE  = re.compile(r"crossover[_ -]*rate[ \t=:]+([0-9.]+)\b", re.I)
RGX_MSTRAT = re.compile(r"mutation[_ -]*strategy[ \t=:]+([A-Za-z0-9_.+-]+)", re.I)
RGX_MRATE  = re.compile(r"mutation[_ -]*rate[ \t=:]+([0-9.]+)\b", re.I)
RGX_GEN    = re.compile(r"generational[ \t=:]+(True|False)", re.I)
RGX_EVAL_D = re.compile(r"eval duration:\s*([0-9.]+)")

# Abbreviations
X_ABBR = {
    "constrained_smith_waterman": "CSWX",
    "shortest_edit_path": "SEPX",
    "one_point": "1PX",
    "None": "None",
    "": "None",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_text(s:str)->str:
    return RGX_ANSI.sub("", s).replace("\r", "\n")

def secs_to_hm(s: float) -> str:
    s = int(math.floor(s))
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}:{m:02d}"

def trim_zero(x: str) -> str:
    return x[:-2] if x.endswith(".0") else x

def prob_w3(x: str) -> str:
    """Right-align probability string to width 3: '  0', '0.5', '  1'."""
    return f"{x:>3}"

# ── Parsing ────────────────────────────────────────────────────────────────────
def extract_from_log(p:Path):
    t = clean_text(p.read_text(errors="ignore"))

    m = RGX_SEED1.search(t) or RGX_SEED2.search(t)
    seed = int(m.group(1)) if m else -1

    m = RGX_DATA.search(t); dataset = m.group(1) if m else ""

    xstrat  = (RGX_XSTRAT.findall(t) or [""])[-1]
    xrate   = (RGX_XRATE.findall(t)  or [""])[-1]
    mstrat  = (RGX_MSTRAT.findall(t) or [""])[-1]
    mrate   = (RGX_MRATE.findall(t)  or [""])[-1]
    if xrate in {"0","0.0"}:
        xstrat = "None"
    xabbr = X_ABBR.get(xstrat, xstrat)
    xrate_disp = trim_zero(xrate or "0")
    mut_disp   = trim_zero(mrate or "0")

    m = RGX_GEN.search(t)
    mode = "Generational" if (m and m.group(1).lower()=="true") else "Steady-State"

    lines = t.splitlines()
    both = [ln for ln in lines if RGX_PAIR.search(ln) and RGX_TIME.search(ln)]
    if both:
        m = RGX_PAIR.search(both[-1]); value = int(m.group(1)) if m else None
    else:
        pairs = RGX_PAIR.findall(t); value = int(pairs[-1][0]) if pairs else None

    eval_secs = sum(float(x) for x in RGX_EVAL_D.findall(t)) if "eval duration" in t else 0.0

    row_raw = (mode, xabbr, xrate_disp, mut_disp)  # keep tuple to format later

    return {
        "seed": seed, "dataset": dataset,
        "value": value, "runtime": eval_secs,
        "mode": mode, "xabbr": xabbr, "xrate_disp": xrate_disp, "mut_disp": mut_disp,
        "row_key": row_raw
    }

# ── Row label formatting ───────────────────────────────────────────────────────
def format_row_label(mode: str, xabbr: str, xrate: str, mut: str) -> str:
    # mode fixed 12, two spaces; xabbr right-aligned 4; (p=___) where prob is width 3; two spaces; mut
    return f"{mode:<12}  {xabbr:>4}(p={xrate})  mut={mut}"

def build_row_labels(meta_df: pd.DataFrame) -> dict:
    # meta_df columns: row_key (tuple), mode, xabbr, xrate_disp, mut_disp
    labels = {}
    for _, r in meta_df.iterrows():
        k = r["row_key"]
        labels[k] = format_row_label(r["mode"], r["xabbr"], r["xrate_disp"], r["mut_disp"])
    return labels

# ── Cell formatting ────────────────────────────────────────────────────────────
def fmt_console_cell(val, rt):
    if not rt or rt <= 0:
        return ""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        l1 = "[grey50]⧖[/]"
    else:
        try:
            n = int(val)
            l1 = f"[bold green]{n}[/]" if n >= 1000 else str(n)
        except Exception:
            l1 = str(val)
    l2 = f"[grey50]⧗ {secs_to_hm(rt)}[/]"
    return f"{l1}\n{l2}"

def fmt_md_cell(val, rt):
    if not rt or rt <= 0:
        return ""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        l1 = "⧖"
    else:
        try:
            n = int(val)
            l1 = f"**{n}**" if n >= 1000 else str(n)
        except Exception:
            l1 = str(val)
    l2 = f"⧗ {secs_to_hm(rt)}"
    return f"{l1}<br>{l2}"

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Directory with *.txt logs (searched recursively)")
    ap.add_argument("--md", action="store_true", help="Print Markdown tables")
    ap.add_argument("--sort-rows", choices=["asc","desc"], default="asc")
    ap.add_argument("--sort-cols", choices=["asc","desc"], default="asc")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Not found: {root}", file=sys.stderr); sys.exit(2)

    recs = []
    for p in sorted(root.rglob("*.txt")):
        try:
            r = extract_from_log(p)
            if r["dataset"] and r["value"] is not None:
                recs.append(r)
        except Exception:
            pass
    if not recs:
        print("No usable records.", file=sys.stderr); sys.exit(1)

    df = pd.DataFrame(recs)
    seeds = sorted(df["seed"].unique())

    if args.md:
        for s in seeds:
            sub = df[df.seed == s]
            pt_val = pd.pivot_table(sub, index="row_key", columns="dataset", values="value",   aggfunc="max")
            pt_rt  = pd.pivot_table(sub, index="row_key", columns="dataset", values="runtime", aggfunc="max")
            pt_val = pt_val.sort_index(ascending=(args.sort_rows=="asc"))
            cols   = sorted(pt_val.columns, reverse=(args.sort_cols=="desc"))
            pt_val = pt_val.reindex(cols, axis=1)
            pt_rt  = pt_rt.reindex(index=pt_val.index, columns=pt_val.columns)

            meta = sub[["row_key","mode","xabbr","xrate_disp","mut_disp"]].drop_duplicates("row_key")
            rename_map = {k: format_row_label(m, x, xr, mu) for k,m,x,xr,mu in meta[["row_key","mode","xabbr","xrate_disp","mut_disp"]].itertuples(index=False, name=None)}
            pt_val.rename(index=rename_map, inplace=True)
            pt_rt  = pt_rt.rename(index=rename_map)

            out = pt_val.astype(object)
            for r in out.index:
                for c in out.columns:
                    v  = pt_val.at[r, c] if c in pt_val.columns else None
                    rt = float(pt_rt.at[r, c]) if (c in pt_rt.columns and pd.notna(pt_rt.at[r, c])) else 0.0
                    out.at[r, c] = fmt_md_cell(v, rt)

            colalign = ["left"] + ["right"] * out.shape[1]
            print(f"\n### Seed {s}\n")
            print(out.to_markdown(colalign=colalign))
    else:
        console = Console()
        for s in seeds:
            sub = df[df.seed == s]
            pt_val = pd.pivot_table(sub, index="row_key", columns="dataset", values="value",   aggfunc="max")
            pt_rt  = pd.pivot_table(sub, index="row_key", columns="dataset", values="runtime", aggfunc="sum")
            pt_val = pt_val.sort_index(ascending=(args.sort_rows=="asc"))
            cols   = sorted(pt_val.columns, reverse=(args.sort_cols=="desc"))
            pt_val = pt_val.reindex(cols, axis=1)
            pt_rt  = pt_rt.reindex(index=pt_val.index, columns=pt_val.columns)

            console.rule(f"Seed {s}", align="left")
            table = Table(show_header=True, header_style="bold", show_lines=False, pad_edge=False)
            table.add_column("row", justify="left", no_wrap=True)
            for col in pt_val.columns:
                table.add_column(str(col), justify="right", no_wrap=True)

            for idx in pt_val.index:
                mode, xabbr, xrate, mut = idx  # row_key tuple
                row_label = format_row_label(mode, xabbr, xrate, mut)
                cells = []
                for col in pt_val.columns:
                    v  = pt_val.at[idx, col] if col in pt_val.columns else None
                    rt = float(pt_rt.at[idx, col]) if (col in pt_rt.columns and pd.notna(pt_rt.at[idx, col])) else 0.0
                    cells.append(fmt_console_cell(v, rt))
                table.add_row(row_label, *cells)

            console.print(table)

if __name__ == "__main__":
    main()

