#!/usr/bin/env python3
import argparse
import json
import os
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import thread_time

import cloudpickle as pickle
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_strategies.utils.recursive_constrained_smith_waterman import (
    recursive_constrained_smith_waterman_crossover,
)

def save_json(file_path, obj) -> None:
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def extract_dataset(p: str) -> str:
    return Path(p).name.split("_")[0]

def extract_values(p: str) -> list[str]:
    return Path(p).stem.split("_")[:-1]

def target_basename(a: str, b: str) -> str:
    return "_".join(extract_values(a) + extract_values(b))

def worker(a: str, b: str, results_dir: Path, logs_dir: Path, skip_existing: bool = True):
    base = target_basename(a, b)
    out_path = results_dir / f"{base}.json"
    log_path = logs_dir / f"{base}.log"
    if skip_existing and out_path.exists():
        return ("skip", str(out_path))

    with log_path.open("w", encoding="utf-8") as log:
        def logp(msg: str) -> None:
            log.write(msg + "\n")

        try:
            logp(f"Loading A: {a}")
            with open(a, "rb") as fh:
                dt_a = pickle.load(fh)[0]
            logp(f"Loading B: {b}")
            with open(b, "rb") as fh:
                dt_b = pickle.load(fh)[0]

            size_a = len(dt_a.serialise())
            size_b = len(dt_b.serialise())
            logp(f"Sizes — A:{size_a} B:{size_b}")

            logp("Computing distance...")
            t0 = thread_time()
            _, _, _, _, _, dist = recursive_constrained_smith_waterman_crossover(dt_a, dt_b)
            td = thread_time() - t0
            logp(f"Done in {td:.6f} s")

            rec = {
                "dataset": extract_dataset(a),
                "size_a": size_a,
                "size_b": size_b,
                "dist": dist,
                "time": td,
            }
            save_json(out_path, rec)
            logp(f"Saved: {out_path}")
            return ("ok", str(out_path))
        except Exception as e:
            err = {"error": type(e).__name__, "message": str(e)}
            save_json(out_path.with_suffix(".error.json"), err)
            logp(f"ERROR: {err}")
            return ("err", str(out_path))

def main():
    ap = argparse.ArgumentParser(
        description="Compute pairwise distances from CSV (space-separated columns a b)."
    )
    ap.add_argument("csv", help="Path to CSV with pairs: a<space>b")
    ap.add_argument("--workers", type=int, default=min(32, (os.cpu_count() * 10 or 4) + 4))
    ap.add_argument("--results-dir", type=Path, default=Path("results/distance"))
    ap.add_argument("--logs-dir", type=Path, default=Path("logs/distance"))
    ap.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                    help="Skip pairs with existing JSON result (default: True)")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="Recompute even if JSON exists")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, sep=" ", names=["a", "b"], dtype=str)
    if df.empty:
        print("No pairs found.")
        return

    # Optional dataset consistency check on column 'a'
    if df["a"].map(extract_dataset).nunique() != 1:
        raise ValueError("Not all datasets identical in column 'a'.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(df.itertuples(index=False, name=None))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, a, b, args.results_dir, args.logs_dir, args.skip_existing)
                for a, b in pairs]
        done_ok = done_skip = done_err = 0
        for f in as_completed(futs):
            status, _ = f.result()
            if status == "ok":
                done_ok += 1
            elif status == "skip":
                done_skip += 1
            else:
                done_err += 1
        print(f"OK:{done_ok} SKIP:{done_skip} ERR:{done_err}")

if __name__ == "__main__":
    main()
