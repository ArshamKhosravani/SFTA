# prepare_eclipse_messages.py
import os
import argparse
import json
import csv
import pandas as pd
from typing import Tuple
from config_messages import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

DEFAULT_START = "2007-11-01"
DEFAULT_END   = "2015-11-30"
DEFAULT_TARGET = 16106

def read_csv_with_fallback(path: str, encoding="utf-8") -> Tuple[pd.DataFrame, str]:
    try:
        return pd.read_csv(path, encoding=encoding, low_memory=False), "utf-8"
    except UnicodeDecodeError:
        print(f"Warning: could not decode {path!r} with {encoding}, retrying latin-1")
        return pd.read_csv(path, encoding="latin-1", low_memory=False), "latin-1"

def ensure_datetime_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"CSV must contain '{col}'.")
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df = df[df[col].notna()].copy()
    return df

def filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    df = ensure_datetime_col(df, "creation_time")
    s = pd.Timestamp(start + "T00:00:00Z")
    e = pd.Timestamp(end   + "T23:59:59Z")
    return df[(df["creation_time"] >= s) & (df["creation_time"] <= e)].copy()

def add_assignee_column(df: pd.DataFrame) -> pd.DataFrame:
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

def _str_or_empty(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def _coalesce_text(*candidates) -> str:
    for c in candidates:
        s = _str_or_empty(c).strip()
        if s:
            return s
    return ""  # allow empty

def to_jsonl(rows_df: pd.DataFrame, out_path: str, system_prompt: str, user_tmpl: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, r in rows_df.iterrows():
            title = _coalesce_text(r.get("summary_update"), r.get("summary"))
            body  = _coalesce_text(r.get("description_update"), r.get("description"))
            assignee = _str_or_empty(r.get("assignee"))
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_tmpl.format(title=title, body=body)},
                {"role": "assistant", "content": assignee},
            ]
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows_df)} examples to {out_path}")

def to_title_body_assignee_csv(rows_df: pd.DataFrame, out_path: str):
    """
    Write CSV with exactly: title,body,assignee (in that order).
    - Data fields are fully quoted (QUOTE_ALL), preserving newlines in body.
    - Header is written WITHOUT quotes to match your standard.
    - UTF-8 with BOM for Excel-friendliness on Windows.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    title_series = rows_df.apply(
        lambda r: _coalesce_text(r.get("summary_update"), r.get("summary")), axis=1
    )
    body_series = rows_df.apply(
        lambda r: _coalesce_text(r.get("description_update"), r.get("description")), axis=1
    )
    if "assignee" in rows_df.columns:
        assignee_series = rows_df["assignee"].map(_str_or_empty)
    else:
        assignee_series = pd.Series([""] * len(rows_df), index=rows_df.index)

    out_df = pd.DataFrame(
        {"title": title_series, "body": body_series, "assignee": assignee_series},
        columns=["title", "body", "assignee"]
    )


    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("title,body,assignee\n")
        out_df.to_csv(f, index=False, header=False, quoting=csv.QUOTE_ALL)
    print(f"Wrote {len(out_df)} rows to {out_path} (unquoted header; QUOTE_ALL data)")

def main():
    ap = argparse.ArgumentParser(
        description="Eclipse (Nov 2007–Nov 2015): random 80/10/10; train/valid as JSONL, test as CSV (keep ALL rows)."
    )
    ap.add_argument("--eclipse_csv", default="eclipse_my_augmented.csv",
                    help="Path to Eclipse CSV (default tries eclipse_my_augmented.csv, else eclipse_my.csv)")
    ap.add_argument("--out_dir", required=True,
                    help="Output dir for train_all.jsonl, valid_all.jsonl, test_all.csv")
    ap.add_argument("--seed", type=int, default=3407, help="Random seed for split")
    ap.add_argument("--encoding", default="utf-8", help="Preferred CSV encoding")
    ap.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD), inclusive")
    ap.add_argument("--end",   default=DEFAULT_END,   help="End date (YYYY-MM-DD), inclusive")
    ap.add_argument("--target_exact", type=int, default=DEFAULT_TARGET,
                    help="Trim/keep oldest-first to exactly this many in-window rows (default 16106).")
    ap.add_argument("--allow_below_target", action="store_true",
                    help="If fewer rows than target in window, continue anyway (no error).")
    args = ap.parse_args()

    csv_path = args.eclipse_csv
    if not os.path.exists(csv_path):
        fallback = "eclipse_my.csv"
        if os.path.exists(fallback):
            print(f"→ {csv_path} not found; using {fallback}")
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Neither {args.eclipse_csv} nor eclipse_my.csv exists.")

    print(f"→ Loading CSV: {os.path.abspath(csv_path)}")
    df_raw, enc = read_csv_with_fallback(csv_path, args.encoding)

    df_raw = ensure_datetime_col(df_raw, "creation_time")
    df_span = filter_window(df_raw, args.start, args.end)

    n_before = len(df_span)
    if args.target_exact and n_before >= args.target_exact:
        df_span = df_span.sort_values("creation_time").head(args.target_exact).copy()
        print(f"  In-window rows: {n_before:,} → trimmed to exact {len(df_span):,}")
    elif args.target_exact and n_before < args.target_exact:
        msg = (f"Only {n_before:,} rows in window (< target {args.target_exact:,}). "
               f"{'Continuing' if args.allow_below_target else 'Re-run after augmenting or use --allow_below_target'}.")
        if args.allow_below_target:
            print("  " + msg)
        else:
            raise ValueError(msg)

    df_span = add_assignee_column(df_span)

    # split 80/10/10
    n = len(df_span)
    df_shuf = df_span.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)
    n_test  = n - n_train - n_val

    train_df = df_shuf.iloc[:n_train].copy()
    val_df   = df_shuf.iloc[n_train:n_train + n_val].copy()
    test_df  = df_shuf.iloc[n_train + n_val:].copy()

    os.makedirs(args.out_dir, exist_ok=True)

    to_jsonl(train_df, os.path.join(args.out_dir, "train_all.jsonl"), SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
    to_jsonl(val_df,   os.path.join(args.out_dir, "valid_all.jsonl"), SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

    test_csv_path = os.path.join(args.out_dir, "test_all.csv")
    to_title_body_assignee_csv(test_df, test_csv_path)

    print(f"SUMMARY: total_in_window={n}  train={len(train_df)}  valid={len(val_df)}  test={len(test_df)}")
    print(f"Span: {args.start} .. {args.end}  |  Source CSV: {csv_path} (encoding={enc})")

if __name__ == "__main__":
    main()
