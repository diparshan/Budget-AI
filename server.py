# server.py — Flask UI with canonical category colors and user category correction

import os
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename

# Your predictor (unchanged)
from predict_category import predict_category_for_walmart_dl

# ---- Paths / config ----
TOKENIZER = "preprocessors/tokenizer.pkl"
LABEL_ENCODER = "preprocessors/label_encoder_dl.pkl"
MODEL = "models/bidirectional_lstm_model.keras"

DATA_DIR = "data"
PRED_DIR = "predictions"
CORR_DIR = "corrected"           # corrected CSVs live here
ALLOWED_EXTENSIONS = {"pdf"}

# Ensure folders exist at startup
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CORR_DIR, exist_ok=True)

app = Flask(__name__)

# ---- Canonical category names (your official set) ----
# Any raw/predicted label will be mapped to these exact labels.
CANON_MAP = {
    "entertainment and recreation": "Entertainment and Recreation",
    "foreign currency transactions": "Foreign Currency Transactions",
    "health and education": "Health and Education",
    "other transactions": "Other Transactions",
    "personal and household expenses": "Personal and Household Expenses",
    "professional and financial services": "Professional and Financial Services",
    "restaurants": "Restaurants",
    "retail and grocery": "Retail and Grocery",
    "transportation": "Transportation",
}

def _canon(cat: str) -> str:
    """Map any free-text category to our canonical label."""
    if not isinstance(cat, str):
        return "Other Transactions"
    key = cat.strip().lower()
    return CANON_MAP.get(key, cat.strip())

# ---- Category color theme (keys MUST match canonical labels above) ----
CATEGORY_PALETTE = {
    "Restaurants": "#f97316",                           # orange
    "Retail and Grocery": "#22c55e",                    # green
    "Personal and Household Expenses": "#a855f7",       # purple
    "Professional and Financial Services": "#06b6d4",   # cyan
    "Transportation": "#eab308",                        # yellow
    "Entertainment and Recreation": "#ef4444",          # red
    "Health and Education": "#3b82f6",                  # blue
    "Foreign Currency Transactions": "#8b5cf6",         # violet
    "Other Transactions": "#64748b",                    # slate
}

DEFAULT_COLORS = [
    "#ef4444", "#f59e0b", "#22c55e", "#06b6d4", "#3b82f6", "#8b5cf6",
    "#a855f7", "#eab308", "#10b981", "#f97316", "#db2777", "#64748b",
]

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/health")
def health():
    """Lightweight readiness check (files present, can write predictions/corrected)."""
    errors = []
    for p in (TOKENIZER, LABEL_ENCODER, MODEL):
        if not os.path.exists(p):
            errors.append(f"missing:{p}")
    # Can write to pred/corrected
    try:
        os.makedirs(PRED_DIR, exist_ok=True)
        t1 = os.path.join(PRED_DIR, ".healthcheck")
        with open(t1, "w") as f:
            f.write("ok")
        os.remove(t1)
    except Exception as e:
        errors.append(f"pred_dir:{e}")
    try:
        os.makedirs(CORR_DIR, exist_ok=True)
        t2 = os.path.join(CORR_DIR, ".healthcheck")
        with open(t2, "w") as f:
            f.write("ok")
        os.remove(t2)
    except Exception as e:
        errors.append(f"corr_dir:{e}")
    status = 200 if not errors else 500
    return {"ok": not errors, "errors": errors}, status

@app.get("/")
def index():
    return render_template("index.html")

def _summarize(df: pd.DataFrame) -> dict:
    """
    Build the chart/table summary using per-row fallback:
    User_Corrected_Category (if non-empty) -> Predicted Category -> Category.
    This avoids NaN issues and doesn't require 'all-or-nothing' corrected usage.
    """
    dff = df.copy()

    # Base category: predicted if present else raw
    base_col = "Predicted Category" if "Predicted Category" in dff.columns else "Category"
    if base_col not in dff.columns:
        # If neither is present, create empty column to avoid KeyError
        dff[base_col] = ""

    # Ensure strings, not NaN
    dff[base_col] = dff[base_col].astype("string").fillna("").str.strip()

    # Corrected column cleaned (may not exist yet)
    if "User_Corrected_Category" not in dff.columns:
        dff["User_Corrected_Category"] = ""
    dff["User_Corrected_Category"] = (
        dff["User_Corrected_Category"].astype("string").fillna("").str.strip()
    )

    # Effective per-row category: corrected if non-empty, else base
    dff["__cat"] = dff["User_Corrected_Category"].where(
        dff["User_Corrected_Category"] != "", dff[base_col]
    )
    # Canonicalize labels
    dff["__cat"] = dff["__cat"].map(lambda x: _canon(x))

    # Amount numeric
    dff["Amount"] = pd.to_numeric(dff.get("Amount", 0), errors="coerce").fillna(0.0)

    # Aggregate
    summary_series = (
        dff.groupby("__cat")["Amount"]
           .sum()
           .sort_values(ascending=False)
    )

    labels = summary_series.index.tolist()
    values = [round(float(v), 2) for v in summary_series.values]
    colors = [CATEGORY_PALETTE.get(lab, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
              for i, lab in enumerate(labels)]

    series = [{"label": lab, "value": val, "color": col}
              for lab, val, col in zip(labels, values, colors)]

    color_map = {lab: col for lab, col in zip(labels, colors)}
    total = round(sum(values), 2)

    return {
        "labels": labels,
        "values": values,
        "colors": colors,
        "series": series,
        "color_map": color_map,
        "total": total,
        "month": "All",
    }


@app.post("/upload")
def upload_pdf():
    """Accept a PDF, run prediction, save predictions CSV, show results."""
    if "statement" not in request.files:
        return Response(
            "<p>Error: field <code>statement</code> is required.</p><p><a href='/'>Go back</a></p>",
            mimetype="text/html", status=400
        )
    f = request.files["statement"]
    if not f or not _allowed(f.filename):
        return Response(
            "<p>Error: please upload a .pdf file.</p><p><a href='/'>Go back</a></p>",
            mimetype="text/html", status=400
        )

    safe_name = secure_filename(f.filename)
    in_path = os.path.join(DATA_DIR, safe_name)
    f.save(in_path)

    try:
        df = predict_category_for_walmart_dl(in_path, TOKENIZER, LABEL_ENCODER, MODEL)
    except Exception as e:
        return Response(
            f"<p>Prediction failed: {e}</p><p><a href='/'>Try again</a></p>",
            mimetype="text/html", status=500
        )

    if df is None or df.empty:
        return Response(
            "<p>No transactions found in the PDF.</p><p><a href='/'>Upload another file</a></p>",
            mimetype="text/html"
        )

    # Normalize category labels to your canonical set
    cat_col = "Predicted Category" if "Predicted Category" in df.columns else "Category"
    df[cat_col] = df[cat_col].astype(str).apply(_canon)

    # Ensure Amount is numeric (so Jinja '{:,.2f}' works)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

    # Save predictions CSV
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_csv = os.path.join(PRED_DIR, f"{base}_predictions.csv")
    df.to_csv(out_csv, index=False)
    download_name = os.path.basename(out_csv)

    # Build summary & preview
    summary = _summarize(df)
    preview = df.head(50).to_dict(orient="records")

    return render_template(
        "results.html",
        rows=len(df),
        download_name=download_name,
        download_kind="predictions",     # tell the template which download route to use
        summary=summary,
        preview=preview
    )

# ---- Downloads ----
@app.get("/predictions/<path:filename>")
def download_prediction(filename):
    return send_from_directory(PRED_DIR, filename, as_attachment=True)

@app.get("/corrected/<path:filename>")
def download_corrected(filename):
    return send_from_directory(CORR_DIR, filename, as_attachment=True)

# ---- Editing categories ----
@app.get("/edit/<path:filename>")
def edit_categories(filename):
    """
    Render the editor for per-row 'User_Corrected_Category'.
    Accepts either a predictions filename (e.g., 'view_predictions.csv')
    OR a corrected filename (e.g., 'view_predictions_corrected.csv').
    Prefers corrected if it exists, else falls back to predictions.
    """
    basename = os.path.basename(filename)
    name_no_ext, _ = os.path.splitext(basename)

    # If user passed a corrected file, use it directly; otherwise prefer corrected if present
    if name_no_ext.endswith("_corrected"):
        # exact corrected file as given
        corrected_path = os.path.join(CORR_DIR, basename)
        # compute the corresponding predictions base (strip the suffix)
        pred_base = name_no_ext[:-10]  # remove "_corrected"
        pred_path = os.path.join(PRED_DIR, f"{pred_base}.csv")
        path = corrected_path if os.path.exists(corrected_path) else pred_path
    else:
        # predictions file as given
        pred_path = os.path.join(PRED_DIR, basename)
        corrected_path = os.path.join(CORR_DIR, f"{name_no_ext}_corrected.csv")
        path = corrected_path if os.path.exists(corrected_path) else pred_path

    if not os.path.exists(path):
        return Response(f"<p>File not found: {basename}</p><p><a href='/'>Go back</a></p>",
                        mimetype="text/html", status=404)

    df = pd.read_csv(path)

    # Ensure expected columns exist and are clean (no NaN shown in form)
    if "Predicted Category" not in df.columns and "Category" in df.columns:
        df["Predicted Category"] = df["Category"]
    if "User_Corrected_Category" not in df.columns:
        df["User_Corrected_Category"] = ""

    for col in ["User_Corrected_Category", "Predicted Category", "Category",
                "Trans Date", "Post Date", "Description"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").str.strip()

    if "Amount" in df.columns:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

    options = list(CANON_MAP.values())
    rows = df.to_dict(orient="records")

    return render_template(
        "edit.html",
        filename=basename,   # keeps your existing hidden input working
        rows=rows,
        options=options
    )



def _pred_base_from_source(source: str) -> str:
    """Return predictions base name without extension and without trailing '_corrected' if present."""
    name = os.path.splitext(os.path.basename(source))[0]
    if name.endswith("_corrected"):
        name = name[:-10]
    return name

@app.post("/save_corrections")
def save_corrections():
    """
    Save per-row 'User_Corrected_Category' only for rows the user changed.
    Show Results with a 'User_Corrected_Category' column,
    but export the download WITHOUT that column (final Category only).
    """
    source = request.form.get("source_csv", "").strip()
    if not source:
        return Response("Missing source_csv", mimetype="text/plain", status=400)

    pred_base = _pred_base_from_source(source)
    pred_path = os.path.join(PRED_DIR, f"{pred_base}.csv")
    corrected_path = os.path.join(CORR_DIR, f"{pred_base}_corrected.csv")

    # Load corrected if exists (preserve past edits), else predictions
    load_path = corrected_path if os.path.exists(corrected_path) else pred_path
    if not os.path.exists(load_path):
        return Response(f"Source CSV not found: {load_path}",
                        mimetype="text/plain", status=404)

    df = pd.read_csv(load_path)

    # Ensure columns exist & clean NaN -> ""
    if "User_Corrected_Category" not in df.columns:
        df["User_Corrected_Category"] = ""
    df["User_Corrected_Category"] = df["User_Corrected_Category"].astype("string").fillna("").str.strip()

    if "Predicted Category" not in df.columns and "Category" in df.columns:
        df["Predicted Category"] = df["Category"]
    if "Predicted Category" in df.columns:
        df["Predicted Category"] = df["Predicted Category"].astype("string").fillna("").str.strip()

    # number of rows submitted
    try:
        n = int(request.form.get("n", str(len(df))))
    except ValueError:
        n = len(df)

    # Update ONLY rows that changed (non-empty selection)
    for i in range(min(n, len(df))):
        val = request.form.get(f"cat_{i}", None)
        if val is not None and val.strip() != "":
            df.loc[i, "User_Corrected_Category"] = _canon(val.strip())

    # ---------- Build FINAL 'Category' for preview & summary ----------
    # base = Category (if present) else Predicted Category
    if "Category" in df.columns:
        base = df["Category"].astype("string").fillna("").str.strip()
    else:
        base = pd.Series([""] * len(df), dtype="string")

    pred = df["Predicted Category"].astype("string").fillna("").str.strip() if "Predicted Category" in df.columns else pd.Series([""] * len(df), dtype="string")
    corr = df["User_Corrected_Category"].astype("string").fillna("").str.strip()

    # effective category: corrected -> base -> predicted
    effective = corr.where(corr != "", base.where(base != "", pred)).map(_canon)

    # Preview dataframe keeps User_Corrected_Category visible
    df_view = df.copy()
    df_view["Category"] = effective
    df_view["Amount"] = pd.to_numeric(df_view.get("Amount", 0), errors="coerce").fillna(0.0)

    # Download/export dataframe drops User_Corrected_Category (final result)
    df_export = pd.DataFrame({
        "Trans Date": df_view.get("Trans Date", ""),
        "Post Date":  df_view.get("Post Date", ""),
        "Description": df_view.get("Description", ""),
        "Category":   df_view["Category"],
        "Amount":     df_view["Amount"]
    })

    out_name = f"{pred_base}_corrected.csv"
    out_path = os.path.join(CORR_DIR, out_name)
    df_export.to_csv(out_path, index=False)

    # Render results using df_view (so the page can show User_Corrected_Category)
    summary = _summarize(df_view)
    preview = df_view.head(50).to_dict(orient="records")
    any_corrected = (
    df_view["User_Corrected_Category"].astype(str).str.strip().ne("").any()
    if "User_Corrected_Category" in df_view.columns else False
)

    return render_template(
        "results.html",
        rows=len(df_view),
        download_name=out_name,
        download_kind="corrected",    # template chooses /corrected route
        summary=summary,
        preview=preview,
        any_corrected=any_corrected   # <-- tells template to show corrected column
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
