#!/usr/bin/env python3
import os
import io
import sys
import zipfile
import pandas as pd

# ---------- helpers ----------

def _norm(s: str) -> str:
    """Normalize a string for lenient column matching (case/spacing/underscore/hyphen-insensitive)."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def _get_column(df: pd.DataFrame, *candidates) -> str | None:
    """Return the first column in df that matches any candidate name (lenient matching)."""
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None

def _get_first_present(df: pd.DataFrame, lists_of_candidates: list[list[str]]) -> str | None:
    """Try lists of synonyms in order; return the first actual column name found."""
    for group in lists_of_candidates:
        col = _get_column(df, *group)
        if col:
            return col
    return None

def _read_csv_robust_any(*, path: str | None = None, data: bytes | None = None, label: str = "") -> pd.DataFrame:
    """
    Robust CSV reader with encoding + delimiter sniffing.
    Accepts a file path or raw bytes.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    seps = (None, ",", ";", "\t", "|")  # None => auto-sniff (python engine)
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                if data is not None:
                    buf = io.BytesIO(data)
                    df = pd.read_csv(buf, encoding=enc, engine="python", sep=sep, on_bad_lines="warn")
                else:
                    df = pd.read_csv(path, encoding=enc, engine="python", sep=sep, on_bad_lines="warn")
                print(f"Loaded '{label or (os.path.basename(path) if path else 'in-memory')}' as CSV with encoding={enc}, sep={'auto' if sep is None else repr(sep)}")
                return df
            except Exception as e:
                last_err = e
                continue
    raise last_err

def _read_table_robust(path: str) -> pd.DataFrame:
    """
    Smart reader:
      - CSV (robust)
      - Excel (xlsx/xlsm/xls)
      - Zip containing CSV
      - Detects Apple Numbers packages and errors with guidance
    """
    lower = path.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        print(f"Loaded '{os.path.basename(path)}' as Excel")
        return pd.read_excel(path)

    # Peek signature to detect ZIP-based formats (xlsx & Numbers are ZIPs)
    with open(path, "rb") as f:
        sig = f.read(4)

    if sig == b"PK\x03\x04":  # ZIP
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            if any(n.startswith("xl/") for n in names):
                print(f"Loaded '{os.path.basename(path)}' as Excel (zip-based .xlsx with wrong extension)")
                return pd.read_excel(path)
            if any(n.startswith("Index/") and n.endswith(".iwa") for n in names) or "Index/Document.iwa" in names:
                raise ValueError(
                    f"File '{path}' is an Apple Numbers document. "
                    "Open in Numbers and export via: File → Export To → CSV, then rerun."
                )
            csv_members = [n for n in names if n.lower().endswith(".csv")]
            if csv_members:
                first_csv = csv_members[0]
                print(f"Detected ZIP; reading inner CSV: {first_csv}")
                with z.open(first_csv) as fh:
                    data = fh.read()
                return _read_csv_robust_any(data=data, label=f"{os.path.basename(path)}::{first_csv}")
        raise ValueError(f"File '{path}' is a ZIP but has no Excel workbook or CSV inside.")

    # Fallback: treat as CSV
    return _read_csv_robust_any(path=path)

def _clean_amount_series(s: pd.Series) -> pd.Series:
    """
    Convert strings like '$1,234.56', '(45.67)', ' 1 234,56 ' to numeric.
    - removes $ and commas
    - parentheses => negative
    - trims spaces
    - handles simple European decimal comma
    """
    if s is None:
        return pd.Series(dtype="float64")
    ss = s.astype(str).str.strip()
    ss = ss.str.replace(r"^\((.*)\)$", r"-\1", regex=True)        # (123.45) -> -123.45
    ss = ss.str.replace(r"[\$,]", "", regex=True)                  # remove $ and ,
    ss = ss.str.replace(r"\s+", "", regex=True)                    # remove spaces
    euro_mask = ss.str.contains(r"^\-?\d+,\d{1,2}$") & ~ss.str.contains(r"\.")
    ss = ss.where(~euro_mask, ss.str.replace(",", ".", regex=False))
    return pd.to_numeric(ss, errors="coerce")

# ---------- core script (append-all) ----------

def update_master_data(corrected_predictions_path_or_pdf: str,
                       master_data_csv: str = "data/combined_transactions_copy.csv") -> None:
    """
    APPEND-ALL MODE:
    - Reads a corrected predictions file (CSV/Excel). If a PDF path is given, derives 'predictions/<base>_dl_for_review.csv'.
    - Resolves Final_Category = User_Corrected_Category > Predicted Category > existing Category.
    - Appends every row with a non-empty Final_Category to the master CSV.
    - Never overwrites existing rows.
    """
    def _resolve_corrected_csv(arg_path: str) -> str:
        if arg_path.lower().endswith(".csv") or arg_path.lower().endswith((".xlsx", ".xlsm", ".xls")):
            return arg_path
        base = os.path.splitext(os.path.basename(arg_path))[0]
        return os.path.join("predictions", f"{base}_dl_for_review.csv")

    corrected_path = _resolve_corrected_csv(corrected_predictions_path_or_pdf)

    # Load or initialize master
    if os.path.exists(master_data_csv):
        master_df = _read_table_robust(master_data_csv)
    else:
        print(f"Master not found at '{master_data_csv}'. Creating a new one.")
        master_df = pd.DataFrame(columns=["Trans Date", "Post Date", "Description", "Category", "Amount"])

    if not os.path.exists(corrected_path):
        print(f"Error: Corrected predictions file not found at {corrected_path}")
        return

    corrected_df = _read_table_robust(corrected_path)

    # Find required columns with synonyms
    desc_col = _get_first_present(corrected_df, [["Description", "Desc", "Details", "Narration", "Merchant"]])
    amt_col  = _get_first_present(corrected_df, [["Amount", "Amt", "Value"]])
    pred_col = _get_first_present(corrected_df, [["Predicted Category", "Predicted_Category", "Predicted",
                                                  "Model_Category", "Category_Pred", "DL_Predicted_Category",
                                                  "LR_Predicted_Category"]])
    user_col = _get_first_present(corrected_df, [["User_Corrected_Category", "User Corrected Category",
                                                  "User Corrected", "Corrected_Category", "Correction"]])
    final_existing_col = _get_first_present(corrected_df, [["Final_Category", "Final Category", "Category"]])

    missing = [name for name, col in [("Description", desc_col), ("Amount", amt_col)] if col is None]
    if missing:
        print(f"Error: Missing required column(s) in corrected file: {', '.join(missing)}")
        print(f"Columns available: {list(corrected_df.columns)}")
        return

    # Compute Final_Category (user > predicted > existing Category)
    def _final_cat(row):
        if user_col:
            v = row.get(user_col)
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        if pred_col:
            v = row.get(pred_col, "")
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        if final_existing_col:
            v = row.get(final_existing_col, "")
            if pd.notna(v) and str(v).strip():
                return str(v).strip()
        return ""

    corrected_df["Final_Category"] = corrected_df.apply(_final_cat, axis=1)

    # Ensure master has canonical columns
    for col in ["Trans Date", "Post Date", "Description", "Category", "Amount"]:
        if col not in master_df.columns:
            master_df[col] = pd.NA

    # Clean amount for stable storage
    corrected_df["__amount_num__"] = _clean_amount_series(corrected_df[amt_col])

    # Optional dates
    trans_date_col = _get_first_present(corrected_df, [["Trans Date", "Trans_Date", "Transaction Date", "Date"]])
    post_date_col  = _get_first_present(corrected_df, [["Post Date", "Post_Date"]])

    # Build rows to append (never update/overwrite existing)
    rows = []
    for _, r in corrected_df.iterrows():
        final_cat = r["Final_Category"]
        if not final_cat:  # skip rows with no category at all
            continue
        rows.append({
            "Trans Date": r[trans_date_col] if trans_date_col else pd.NA,
            "Post Date":  r[post_date_col]  if post_date_col  else pd.NA,
            "Description": r[desc_col],
            "Category": final_cat,
            # prefer cleaned numeric amount; if NaN, fall back to original text
            "Amount": r["__amount_num__"] if pd.notna(r["__amount_num__"]) else r[amt_col]
        })

    appended = len(rows)
    if appended:
        master_df = pd.concat([master_df, pd.DataFrame(rows)], ignore_index=True)

    master_df.to_csv(master_data_csv, index=False)
    print(f"[DEBUG] append_all=True | appended={appended} | updated=0")
    print(f"Master data updated successfully from '{os.path.basename(corrected_path)}' into '{master_data_csv}'.")

# ---------- CLI ----------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_master_data.py <corrected_csv_or_pdf> [<master_csv_path>]")
        sys.exit(1)

    input_path = sys.argv[1]
    master_path = sys.argv[2] if len(sys.argv) >= 3 and not sys.argv[2].startswith("-") else "data/combined_transactions_copy.csv"

    update_master_data(input_path, master_path)
