#!/usr/bin/env python3
# scripts/einsearch-progress.py
# Fast-only (head+tail) parser. O(1) RAM per file.
# Usage:
#   scripts/einsearch-progress.py <logs_root1> [<logs_root2> ...] [--md]
#                                 [--sort-rows asc|desc] [--sort-cols asc|desc]
#                                 [--head-kb N] [--tail-kb N]
#                                 [--no-sort]
#                                 [--config-dir DIR] [--config-recursive]
import re, sys, argparse, math, os, shlex, json
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
RGX_XSTRAT2= re.compile(r"\b(?:x?crossover(?:_?strategy)?|xstrat)\s*[:=]\s*([A-Za-z0-9_.+-]+)", re.I)
RGX_XRATE  = re.compile(r"crossover[_ -]*rate[ \t=:]+([0-9.]+)\b", re.I)
RGX_MRATE  = re.compile(r"mutation[_ -]*rate[ \t=:]+([0-9.]+)\b", re.I)
RGX_GEN    = re.compile(r"generational[ \t=:]+(True|False)", re.I)

# ── Abbreviations / display ───────────────────────────────────────────────────
_CANON_ABBR = {
    "recursive_constrained_smith_waterman": "RCSWX",
    "constrained_smith_waterman": "CSWX",
    "shortest_edit_path": "SEPX",
    "one_point": "1PX",
    "none": "None",
}
def _abbr(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "UNK"  # do NOT fold unknown into None
    return _CANON_ABBR.get(s.lower(), s)

# ── Config .lst parsing (schema) ───────────────────────────────────────────────
KNOWN_KEYS = {
    "seed", "dataset", "generational", "regularised", "elitism",
    "mutation_rate", "crossover_strategy", "crossover_rate",
}

def _boolify(v):
    if isinstance(v, bool): return v
    if v is None: return False
    s = str(v).strip().lower()
    if s in {"1","true","yes","on"}:  return True
    if s in {"0","false","no","off"}: return False
    return True  # bare flag

def _cast_known_types(kv: dict) -> dict:
    out = dict(kv)
    if "seed" in out:
        try: out["seed"] = int(out["seed"])
        except Exception: out["seed"] = None
    if "elitism" in out:
        try: out["elitism"] = int(out["elitism"])
        except Exception: out["elitism"] = None
    if "mutation_rate" in out:
        try: out["mutation_rate"] = float(out["mutation_rate"])
        except Exception: out["mutation_rate"] = None
    if "crossover_rate" in out:
        try:
            cr = float(out["crossover_rate"])
            out["crossover_rate"] = int(cr) if float(cr).is_integer() else cr
        except Exception:
            out["crossover_rate"] = None
    if "generational" in out:
        out["generational"] = _boolify(out["generational"])
    if "regularised" in out:
        out["regularised"] = _boolify(out["regularised"])
    # Only force None when explicitly zero
    rate_txt = str(out.get("crossover_rate", "")).strip().lower()
    if rate_txt in {"0", "0.0"}:
        out["crossover_strategy"] = "None"
    return out

def _parse_config_line(s: str):
    """Return base_config:str, kv:dict (supports --k v, --k=v, bare flags)."""
    s = s.strip()
    if not s or s.startswith("#"): return None, {}
    if ";" in s:
        base_config, args = s.split(";", 1)
    else:
        base_config, args = s, ""
    toks = shlex.split(args)
    kv = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t.split("=", 1)
                k = k.lstrip("-").replace("-", "_")
                kv[k] = v; i += 1
            else:
                k = t.lstrip("-").replace("-", "_")
                if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                    kv[k] = toks[i + 1]; i += 2
                else:
                    kv[k] = True; i += 1
        else:
            i += 1
    return base_config.strip(), kv

def parse_config_dir(config_dir: Path, recursive: bool = False):
    it = config_dir.rglob("*.lst") if recursive else config_dir.glob("*.lst")
    lst_files = sorted(p for p in it if p.is_file())
    rows = []
    for p in lst_files:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                base, kv = _parse_config_line(line)
                if base is None: continue
                kv = _cast_known_types(kv)
                extras = {k: v for k, v in kv.items() if k not in KNOWN_KEYS}
                rows.append({
                    "file": str(p.relative_to(config_dir)),
                    "line_no": i,
                    "base_config": base,
                    "dataset": kv.get("dataset"),
                    "seed": kv.get("seed"),
                    "generational": kv.get("generational", False),
                    "regularised": kv.get("regularised", False),
                    "elitism": kv.get("elitism"),
                    "mutation_rate": kv.get("mutation_rate"),
                    "crossover_strategy": kv.get("crossover_strategy"),
                    "crossover_rate": kv.get("crossover_rate"),
                    "extras_json": json.dumps(extras, ensure_ascii=False, sort_keys=True),
                })
    df = pd.DataFrame(rows, columns=[
        "file","line_no","base_config","dataset","seed","generational",
        "regularised","elitism","mutation_rate","crossover_strategy",
        "crossover_rate","extras_json",
    ])
    if not df.empty:
        df["line_no"] = pd.to_numeric(df["line_no"], errors="coerce").astype("Int64")
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
        df["elitism"] = pd.to_numeric(df["elitism"], errors="coerce").astype("Int64")
        df["mutation_rate"] = pd.to_numeric(df["mutation_rate"], errors="coerce")
        df["crossover_rate"] = pd.to_numeric(df["crossover_rate"], errors="coerce")
        df["generational"] = df["generational"].astype("boolean")
        df["regularised"]  = df["regularised"].astype("boolean")
        for c in ("dataset","crossover_strategy","base_config","file"):
            df[c] = df[c].astype("category")
    return df, len(lst_files)

# ── Helpers ────────────────────────────────────────────────────────────────────
def secs_to_hm(s: float) -> str:
    s = int(math.floor(s)); h = s // 3600; m = (s % 3600) // 60
    return f"{h}:{m:02d}"

def trim_zero(x: str) -> str:
    return x[:-2] if isinstance(x, str) and x.endswith(".0") else x

def choose_workers(num_files: int) -> int:
    c = os.cpu_count() or 1
    if c <= 1: n = 1
    elif c < 16: n = max(1, c - 2)
    else: n = 16
    return min(n, num_files)

def _num_to_disp(v, default="0"):
    if v is None or (isinstance(v, float) and pd.isna(v)): return default
    return trim_zero(str(v))

def _last_progress_nums(tail_text: str):
    """Return (num, den) from the last progress-looking line in tail, else (None, None)."""
    last_num, last_den = None, None
    for ln in reversed(tail_text.replace("\r","\n").split("\n")):
        if not ln: continue
        m_pair = RGX_PAIR.search(ln)
        if not m_pair: continue
        if RGX_TIME.search(ln) or last_num is None:
            try:
                last_num, last_den = int(m_pair.group(1)), int(m_pair.group(2))
            except Exception: pass
    return last_num, last_den

def _render_bar(done: float, total: float, width: int = 28, console_mode: bool = True) -> str:
    done = 0.0 if pd.isna(done) else float(done)
    total = 0.0 if pd.isna(total) else float(total)
    pct = 0.0 if total <= 0 else min(1.0, max(0.0, done / total))
    fill = int(round(width * pct))
    return (f"[green]{'█'*fill}[/][grey37]{'░'*(width-fill)}[/]  {int(done)}/{int(total)} ({pct*100:.1f}%)"
            if console_mode else
            f"{'█'*fill}{'░'*(width-fill)}  {int(done)}/{int(total)} ({pct*100:.1f}%)")

# ── Fast path: head + tail ─────────────────────────────────────────────────────
def extract_from_log_fast(p: Path, head_kb: int = 256, tail_kb: int = 1024):
    seed = -1; dataset = ""; xstrat = ""; xrate = ""; mrate = ""; mode = "Steady-State"
    with p.open("rb") as f:
        head = f.read(head_kb * 1024).decode("utf-8", "ignore")
        m = RGX_SEED1.search(head) or RGX_SEED2.search(head);     seed    = int(m.group(1)) if m else -1
        m = RGX_DATA.search(head);                                 dataset = m.group(1) if m else ""
        m = RGX_XSTRAT.search(head) or RGX_XSTRAT2.search(head);   xstrat  = m.group(1) if m else ""
        m = RGX_XRATE.search(head);                                xrate   = m.group(1) if m else ""
        m = RGX_MRATE.search(head);                                mrate   = m.group(1) if m else ""
        m = RGX_GEN.search(head);                                  mode    = "Generational" if (m and m.group(1).lower()=="true") else "Steady-State"
        sz = f.seek(0, os.SEEK_END); start = max(0, sz - tail_kb * 1024); f.seek(start)
        tail = f.read().decode("utf-8", "ignore")
    last_num, last_den = _last_progress_nums(tail)
    xs = (xstrat or "").strip().lower()
    if xs == "none" or xrate in {"0","0.0"}: xabbr = "None"
    elif xs == "":                           xabbr = "UNK"
    else:                                    xabbr = _abbr(xstrat)
    xrate_disp = trim_zero(xrate or "0"); mut_disp = trim_zero(mrate or "0")
    row_raw = (mode, xabbr, xrate_disp, mut_disp)
    return {
        "seed": seed, "dataset": dataset,
        "value": last_num, "target": (last_den if isinstance(last_den, int) else 1000),
        "runtime": 0.0,
        "mode": mode, "xabbr": xabbr, "xrate_disp": xrate_disp, "mut_disp": mut_disp, "row_key": row_raw
    }

# ── Row/cell formatting ────────────────────────────────────────────────────────
def format_row_label(mode: str, xabbr: str, xrate: str, mut: str) -> str:
    return f"{mode:<12}  {xabbr:>5}(p={xrate})  mut={mut}"

def fmt_console_cell(val, rt):
    if val is None or (isinstance(val, float) and pd.isna(val)): l1 = "[grey50]⧖[/]"
    else:
        try:
            n = int(val); l1 = f"[bold green]{n}[/]" if n >= 1000 else str(n)
        except Exception:
            l1 = str(val)
    l2 = f"\n[grey50]⧗ {secs_to_hm(rt)}[/]" if (rt and rt > 0) else ""
    return f"{l1}{l2}"

def fmt_md_cell(val, rt):
    if val is None or (isinstance(val, float) and pd.isna(val)): l1 = "⧖"
    else:
        try:
            n = int(val); l1 = f"**{n}**" if n >= 1000 else str(n)
        except Exception:
            l1 = str(val)
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
    ap.add_argument("--config-dir", default="configs/einspace", help="Directory with *.lst config lists")
    ap.add_argument("--config-recursive", action="store_true", help="Recurse into subdirectories for *.lst")
    args = ap.parse_args()

    # Parse .lst configs → expected cells
    cfg_dir = Path(args.config_dir)
    df_configs, cfg_files = (pd.DataFrame(), 0)
    try:
        if cfg_dir.exists():
            df_configs, cfg_files = parse_config_dir(cfg_dir, args.config_recursive)
            print(f"[cfg] Loaded {len(df_configs)} rows from {cfg_files} .lst files in {cfg_dir} (recursive={args.config_recursive})", file=sys.stderr)
        else:
            print(f"[cfg] Skipping: {cfg_dir} not found", file=sys.stderr)
    except Exception as e:
        print(f"[cfg] Error while parsing {cfg_dir}: {e}", file=sys.stderr)

    # Expected grid (row_key aligned to logs)
    cfg_grid = pd.DataFrame(columns=["seed","dataset","mode","xabbr","xrate_disp","mut_disp","row_key","target"])
    if not df_configs.empty:
        def _mode(b): return "Generational" if bool(b) else "Steady-State"
        cr_raw = df_configs["crossover_rate"]
        strat_eff = df_configs["crossover_strategy"].astype("string")
        strat_eff = strat_eff.mask(cr_raw.fillna("").astype(str).isin(["0","0.0"]), "None")
        tmp = pd.DataFrame({
            "seed": df_configs["seed"].astype("Int64"),
            "dataset": df_configs["dataset"].astype("string"),
            "mode": df_configs["generational"].map(_mode),
            "xabbr": strat_eff.map(_abbr),
            "xrate_disp": df_configs["crossover_rate"].apply(_num_to_disp),
            "mut_disp": df_configs["mutation_rate"].apply(_num_to_disp),
        })
        tmp["row_key"] = list(zip(tmp["mode"], tmp["xabbr"], tmp["xrate_disp"], tmp["mut_disp"]))
        tmp["target"] = 1000
        cfg_grid = tmp.drop_duplicates(subset=["seed","dataset","row_key"], ignore_index=True)

    # Collect logs
    files = []; seen = set()
    def add_file(p: Path):
        try: key = str(p.resolve())
        except Exception: key = str(p)
        if key not in seen and p.is_file():
            seen.add(key); files.append(Path(key))
    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists(): print(f"Not found: {root}", file=sys.stderr); continue
        if root.is_dir():
            for p in root.rglob("*.txt"): add_file(p)
        elif root.is_file() and root.suffix.lower()==".txt":
            add_file(root)
        else:
            print(f"Skipping non-text file: {root}", file=sys.stderr)
    if not files: print("No files.", file=sys.stderr); sys.exit(1)
    if not args.no_sort: files.sort()

    # Parse logs (fast), canonicalize to one row per cell
    recs = []
    parser = (lambda p: extract_from_log_fast(p, args.head_kb, args.tail_kb))
    n_workers = choose_workers(len(files))
    with ThreadPoolExecutor(max_workers=n_workers) as ex, tqdm(total=len(files), desc="Parsing", unit="file") as pbar:
        futs = {ex.submit(parser, p): p for p in files}
        for fu in as_completed(futs):
            try:
                r = fu.result()
                if r["dataset"] and (r["value"] is not None or r["target"]): recs.append(r)
            except Exception: pass
            finally: pbar.update(1)

    df_logs = pd.DataFrame(recs, columns=["seed","dataset","value","target","runtime","mode","xabbr","xrate_disp","mut_disp","row_key"])
    if not df_logs.empty:
        df_logs = (df_logs
            .groupby(["seed","dataset","row_key"], as_index=False)
            .agg(value=("value","max"),
                 target=("target","max"),
                 runtime=("runtime","sum"),
                 mode=("mode","first"),
                 xabbr=("xabbr","first"),
                 xrate_disp=("xrate_disp","first"),
                 mut_disp=("mut_disp","first")))

    # Merge plan with logs for tables (keep ⧖ for missing)
    if not cfg_grid.empty:
        df = cfg_grid.merge(df_logs[["seed","dataset","row_key","value","target","runtime"]],
                            on=["seed","dataset","row_key"], how="left", suffixes=("_cfg",""))
        df["in_cfg"] = True
        df["target"] = df["target"].fillna(df["target_cfg"]).fillna(1000)
        df = df.drop(columns=[c for c in df.columns if c.endswith("_cfg")])
        # (Do not append log-only rows here; tables show plan-only rows.)
    else:
        df = df_logs.copy(); df["in_cfg"] = False
        if "target" not in df.columns: df["target"] = 1000

    if df.empty:
        print("No usable records.", file=sys.stderr); sys.exit(1)

    # ── PROGRESS BARS: targets from plan (.lst); done from logs ∩ plan; normalized per cell ──
    # planned cells (+ method for grouping)
    cfg_cells = cfg_grid[["seed","dataset","row_key","xabbr"]].copy()
    cfg_cells["cell_target"] = 1000

    # observed progress per cell (normalize by denominator when available)
    obs = df_logs[["seed","dataset","row_key","value","target"]].copy() if not df_logs.empty else pd.DataFrame(columns=["seed","dataset","row_key","value","target"])
    if not obs.empty:
        val = obs["value"].fillna(0).astype(float).clip(lower=0)
        den = pd.to_numeric(obs["target"], errors="coerce")
        mask = den.gt(0)
        done_norm = pd.Series(0.0, index=obs.index, dtype=float)
        done_norm.loc[mask] = (val[mask] / den[mask]).clip(upper=1.0) * 1000.0
        done_norm.loc[~mask] = val[~mask].clip(upper=1000.0)
        obs["done_norm"] = done_norm
        obs = obs[["seed","dataset","row_key","done_norm"]]
    else:
        obs = pd.DataFrame(columns=["seed","dataset","row_key","done_norm"])

    # join: only planned cells contribute to bars
    bar = cfg_cells.merge(obs, on=["seed","dataset","row_key"], how="left")
    bar["done_norm"] = bar["done_norm"].fillna(0.0)

    grp_seed    = bar.groupby("seed",    dropna=False).agg(done=("done_norm","sum"), total=("cell_target","sum")).reset_index()
    grp_dataset = bar.groupby("dataset", dropna=False).agg(done=("done_norm","sum"), total=("cell_target","sum")).reset_index()
    grp_method  = bar.groupby("xabbr",   dropna=False).agg(done=("done_norm","sum"), total=("cell_target","sum")).reset_index()

    # Seeds to render
    seeds = sorted([int(s) for s in pd.unique(df["seed"]) if pd.notna(s)])

    # ── Render ─────────────────────────────────────────────────────────────────
    if args.md:
        for s in seeds:
            sub = df[df.seed == s]
            pt_val = pd.pivot_table(sub, index="row_key", columns="dataset", values="value",   aggfunc="max")
            pt_rt  = pd.pivot_table(sub, index="row_key", columns="dataset", values="runtime", aggfunc="sum")
            # Keep plan rows/cols so ⧖ appear
            if not cfg_grid.empty:
                exp = cfg_grid[cfg_grid.seed == s]
                rows_exp = sorted(exp["row_key"].dropna().unique().tolist())
                cols_exp = sorted(exp["dataset"].dropna().unique().tolist(), reverse=(args.sort_cols=="desc"))
                pt_val = pt_val.reindex(index=rows_exp, columns=cols_exp)
                pt_rt  = pt_rt.reindex(index=rows_exp, columns=cols_exp)
                meta_src = exp
            else:
                meta_src = sub

            meta = meta_src[["row_key","mode","xabbr","xrate_disp","mut_disp"]].drop_duplicates("row_key")
            rename_map = {k: format_row_label(m, x, xr, mu)
                          for k, m, x, xr, mu in meta[["row_key","mode","xabbr","xrate_disp","mut_disp"]].itertuples(index=False, name=None)}
            pt_val.rename(index=rename_map, inplace=True)
            pt_rt.rename(index=rename_map, inplace=True)

            if pt_val.empty: continue
            out = pt_val.astype(object)
            for r in out.index:
                for col in out.columns:
                    v  = pt_val.at[r, col] if col in pt_val.columns else None
                    rt_raw = pt_rt.at[r, col] if (col in pt_rt.columns) else None
                    rt = float(rt_raw) if (rt_raw is not None and pd.notna(rt_raw)) else 0.0
                    out.at[r, col] = fmt_md_cell(v, rt)

            colalign = ["left"] + ["right"] * out.shape[1]
            print(f"\n### Seed {s}\n"); print(out.to_markdown(colalign=colalign))

        # Markdown summaries
        def _md_df(g, name_col):
            pct = (g["done"] / g["total"]).fillna(0.0).clip(0,1)
            bars = [_render_bar(d, t, width=28, console_mode=False) for d, t in zip(g["done"], g["total"])]
            return pd.DataFrame({ name_col: g.iloc[:,0], "done": g["done"].round().astype(int),
                                  "total": g["total"].round().astype(int), "percent": (pct*100).round(1), "bar": bars })
        print("\n#### Progress by seed\n");    print(_md_df(grp_seed, "seed").to_markdown(index=False))
        print("\n#### Progress by dataset\n"); print(_md_df(grp_dataset, "dataset").to_markdown(index=False))
        print("\n#### Progress by method\n");  print(_md_df(grp_method.rename(columns={'xabbr':'method'}), "method").to_markdown(index=False))

    else:
        console = Console()
        for s in seeds:
            sub = df[df.seed == s]
            pt_val = pd.pivot_table(sub, index="row_key", columns="dataset", values="value",   aggfunc="max")
            pt_rt  = pd.pivot_table(sub, index="row_key", columns="dataset", values="runtime", aggfunc="sum")
            if pt_val.empty and cfg_grid.empty: continue
            if not cfg_grid.empty:
                exp = cfg_grid[cfg_grid.seed == s]
                rows_exp = sorted(exp["row_key"].dropna().unique().tolist())
                cols_exp = sorted(exp["dataset"].dropna().unique().tolist(), reverse=(args.sort_cols=="desc"))
                pt_val = pt_val.reindex(index=rows_exp, columns=cols_exp)
                pt_rt  = pt_rt.reindex(index=rows_exp, columns=cols_exp)
                meta_src = exp
            else:
                meta_src = sub

            meta = meta_src[["row_key","mode","xabbr","xrate_disp","mut_disp"]].drop_duplicates("row_key")
            rename_map = {k: format_row_label(m, x, xr, mu)
                          for k, m, x, xr, mu in meta[["row_key","mode","xabbr","xrate_disp","mut_disp"]].itertuples(index=False, name=None)}

            pt_val = pt_val.sort_index(ascending=(args.sort_rows == "asc"))
            pt_rt  = pt_rt.reindex(index=pt_val.index, columns=pt_val.columns)

            console.rule(f"Seed {s}", align="left")
            table = Table(show_header=True, header_style="bold", show_lines=False, pad_edge=False)
            table.add_column("row", justify="left", no_wrap=True)
            for col in pt_val.columns: table.add_column(str(col), justify="right", no_wrap=True)

            for idx in pt_val.index:
                mode, xabbr, xrate, mut = idx
                row_label = rename_map.get(idx, format_row_label(mode, xabbr, xrate, mut))
                cells = []
                for col in pt_val.columns:
                    v = pt_val.at[idx, col]
                    rt_raw = pt_rt.at[idx, col] if col in pt_rt.columns else None
                    rt = float(rt_raw) if (rt_raw is not None and pd.notna(rt_raw)) else 0.0
                    cells.append(fmt_console_cell(v, rt))
                table.add_row(row_label, *cells)
            console.print(table)

        # Console progress bars (plan targets; done from logs∩plan)
        console.rule("Progress by seed", align="left")
        for _, r in grp_seed.sort_values("seed").iterrows():
            console.print(f"[bold]{str(r['seed']):>2}[/]  " + _render_bar(r["done"], r["total"], console_mode=True))

        console.rule("Progress by dataset", align="left")
        for _, r in grp_dataset.sort_values("dataset").iterrows():
            name = str(r["dataset"])
            console.print(f"[bold]{name:<12}[/] " + _render_bar(r["done"], r["total"], console_mode=True))

        console.rule("Progress by method", align="left")
        meth_order = ["CSWX","1PX","None","RCSWX","SEPX","UNK"]
        by_method_sorted = grp_method.copy()
        by_method_sorted["_ord"] = by_method_sorted.iloc[:,0].map({m:i for i,m in enumerate(meth_order)}).fillna(99).astype(int)
        by_method_sorted = by_method_sorted.sort_values(["_ord", by_method_sorted.columns[0]]).drop(columns="_ord")
        for _, r in by_method_sorted.iterrows():
            console.print(f"[bold]{str(r['xabbr']):<5}[/] " + _render_bar(r["done"], r["total"], console_mode=True))

if __name__ == "__main__":
    main()

