import pandas as pd
from pathlib import Path


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        ser = df[col]
        dtype = str(ser.dtype)
        non_null = int(ser.notna().sum())
        missing = int(ser.isna().sum())
        missing_pct = float(missing) / len(df) if len(df) > 0 else 0.0
        unique = int(ser.nunique(dropna=True))

        row = {
            "column": col,
            "dtype": dtype,
            "non_null_count": non_null,
            "missing_count": missing,
            "missing_pct": round(missing_pct, 4),
            "unique_values": unique,
        }

        if pd.api.types.is_numeric_dtype(ser):
            desc = ser.describe()
            row.update({
                "mean": None if pd.isna(desc.get("mean")) else float(desc.get("mean")),
                "std": None if pd.isna(desc.get("std")) else float(desc.get("std")),
                "min": None if pd.isna(desc.get("min")) else float(desc.get("min")),
                "25%": None if pd.isna(desc.get("25%")) else float(desc.get("25%")),
                "50%": None if pd.isna(desc.get("50%")) else float(desc.get("50%")),
                "75%": None if pd.isna(desc.get("75%")) else float(desc.get("75%")),
                "max": None if pd.isna(desc.get("max")) else float(desc.get("max")),
            })
        else:
            top = None
            freq = None
            try:
                vc = ser.value_counts(dropna=True)
                if not vc.empty:
                    top = vc.index[0]
                    freq = int(vc.iloc[0])
            except Exception:
                pass
            row.update({"top": top, "top_freq": freq})

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "data" / "processed" / "final_dataset.csv"
    out_csv = repo_root / "results" / "full_dataset_summary.csv"
    out_md = repo_root / "results" / "full_dataset_summary.md"

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    summary = summarize_dataframe(df)

    summary.to_csv(out_csv, index=False)

    # write a simple markdown table (first 200 columns / rows safe)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Full Dataset Summary\n\n")
        f.write(f"Source: {input_path.relative_to(repo_root)}\n\n")
        f.write(summary.to_markdown(index=False))

    print(f"Wrote summary CSV to: {out_csv}")
    print(f"Wrote summary MD to: {out_md}")


if __name__ == "__main__":
    main()
