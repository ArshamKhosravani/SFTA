# prepare_mozilla_messages.py
import os
import argparse
import json
import pandas as pd
from typing import Tuple
from config_messages import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

DEFAULT_START = "1999-06-01"
DEFAULT_END   = "2021-02-28"
DEFAULT_SEED  = 3407
DEFAULT_EXPECT = 110_467 

def read_csv_with_fallback(path: str, encoding="utf-8") -> Tuple[pd.DataFrame, str]:
    try:
        return pd.read_csv(path, encoding=encoding, low_memory=False), "utf-8"
    except UnicodeDecodeError:
        print(f"Warning: could not decode {path!r} with {encoding}, retrying latin-1")
        return pd.read_csv(path, encoding="latin-1", low_memory=False), "latin-1"

def _str_or_empty(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def coalesce(*vals) -> str:
    for v in vals:
        s = _str_or_empty(v).strip()
        if s:
            return s
    return ""

def add_assignee(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "assigned_to_detail.email" in df.columns:
        df["assignee"] = df["assigned_to_detail.email"]
        if "assigned_to" in df.columns:
            df["assignee"] = df["assignee"].fillna(df["assigned_to"])
    elif "assigned_to" in df.columns:
        df["assignee"] = df["assigned_to"]
    else:
        df["assignee"] = ""
    df["assignee"] = df["assignee"].fillna("").astype(str)
    return df

def add_title_body(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # allow empties
    df["title"] = [coalesce(r.get("summary_update"), r.get("summary")) for _, r in df.iterrows()]
    df["body"]  = [coalesce(r.get("description_update"), r.get("description")) for _, r in df.iterrows()]
    return df

def to_jsonl(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(
                    title=_str_or_empty(r["title"]), body=_str_or_empty(r["body"])
                )},
                {"role": "assistant", "content": _str_or_empty(r["assignee"])},
            ]
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(df)} examples to {path}")

def to_csv_slim(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.loc[:, ["title", "body", "assignee"]].to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} rows to {path}")

def main():
    ap = argparse.ArgumentParser(
        description="Mozilla (Jun 1999–Feb 2021): ONLY time-span filter; keep ALL rows (incl. duplicates). "
                    "80/10/10 split. Train/valid JSONL. Test CSV with exactly title,body,assignee."
    )
    ap.add_argument("--mozilla_csv", default="mozilla_my1.csv", help="Mozilla CSV path")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--encoding", default="utf-8", help="Preferred CSV encoding")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    ap.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end",   default=DEFAULT_END,   help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--expect_count", type=int, default=DEFAULT_EXPECT,
                    help="Expected total rows in span (default 110,467)")
    ap.add_argument("--no_strict_count", action="store_true",
                    help="Don’t error if in-span count != expect_count; just warn.")
    args = ap.parse_args()

    if not os.path.exists(args.mozilla_csv):
        raise FileNotFoundError(args.mozilla_csv)

    print(f"→ Loading CSV: {os.path.abspath(args.mozilla_csv)}")
    df, enc = read_csv_with_fallback(args.mozilla_csv, args.encoding)

    if "creation_time" not in df.columns:
        raise KeyError("CSV must contain 'creation_time'.")

    dt = pd.to_datetime(df["creation_time"], errors="coerce", utc=True)
    s = pd.Timestamp(args.start + "T00:00:00Z")
    e = pd.Timestamp(args.end   + "T23:59:59Z")
    mask = dt.notna() & (dt >= s) & (dt <= e)
    df_span = df.loc[mask].copy()
    print(f"  In-span rows (kept): {len(df_span):,}")

    if args.expect_count and not args.no_strict_count and len(df_span) != args.expect_count:
        raise ValueError(f"In-span rows {len(df_span):,} != expected {args.expect_count:,}. "
                         "Rerun with --no_strict_count if you want to proceed anyway.")
    elif args.expect_count and len(df_span) != args.expect_count:
        print(f"  WARNING: In-span rows {len(df_span):,} != expected {args.expect_count:,} (continuing).")

    df_span = add_assignee(df_span)
    df_span = add_title_body(df_span)

    # split 80/10/10
    n = len(df_span)
    df_shuf = df_span.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)
    n_test  = n - n_train - n_val

    train = df_shuf.iloc[:n_train].copy()
    valid = df_shuf.iloc[n_train:n_train+n_val].copy()
    test  = df_shuf.iloc[n_train+n_val:].copy()

    os.makedirs(args.out_dir, exist_ok=True)

    # JSONL for train/valid
    to_jsonl(train, os.path.join(args.out_dir, "train_all.jsonl"))
    to_jsonl(valid, os.path.join(args.out_dir, "valid_all.jsonl"))

    to_csv_slim(test,  os.path.join(args.out_dir, "test_all.csv"))

    print(f"SUMMARY: total_in_span={n:,}  train={len(train):,}  valid={len(valid):,}  test={len(test):,}")
    print(f"Span: {args.start} .. {args.end} | Source: {args.mozilla_csv} (encoding={enc})")

if __name__ == "__main__":
    main()
