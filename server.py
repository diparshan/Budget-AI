# server.py — Flask UI with canonical category colors (reuses your prediction function)
import os
import pandas as pd
from flask import Flask, request, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename

from predict_category import predict_category_for_walmart_dl

# ---- Paths / config ----
TOKENIZER = "preprocessors/tokenizer.pkl"
LABEL_ENCODER = "preprocessors/label_encoder_dl.pkl"
MODEL = "models/bidirectional_lstm_model.keras"

DATA_DIR = "data"
PRED_DIR = "predictions"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)

# ---- Canonical category names (your official set) ----
# Any model output will be mapped to these exact labels.
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
    errors = []
    for p in (TOKENIZER, LABEL_ENCODER, MODEL):
        if not os.path.exists(p):
            errors.append(f"missing:{p}")
    try:
        os.makedirs(PRED_DIR, exist_ok=True)
        testfile = os.path.join(PRED_DIR, ".healthcheck")
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
    except Exception as e:
        errors.append(f"pred_dir:{e}")
    status = 200 if not errors else 500
    return {"ok": not errors, "errors": errors}, status

@app.get("/")
def index():
    return render_template("index.html")

def _category_colors(labels):
    colors, i = [], 0
    for lab in labels:
        colors.append(CATEGORY_PALETTE.get(lab, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]))
        i += 1
    return colors

def _summarize(df: pd.DataFrame) -> dict:
    cat_col = "Predicted Category" if "Predicted Category" in df.columns else "Category"

    # Coerce Amount to numeric for reliable sums
    dff = df.copy()
    dff["Amount"] = pd.to_numeric(dff["Amount"], errors="coerce").fillna(0.0)

    summary_series = (
        dff.groupby(cat_col)["Amount"]
           .sum()
           .sort_values(ascending=False)
    )

    labels = summary_series.index.tolist()
    values = [round(float(v), 2) for v in summary_series.values]

    # Colors aligned to labels
    DEFAULT_COLORS = [
        "#ef4444", "#f59e0b", "#22c55e", "#06b6d4", "#3b82f6", "#8b5cf6",
        "#a855f7", "#eab308", "#10b981", "#f97316", "#db2777", "#64748b",
    ]
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
        "series": series,      # use this in results.html
        "color_map": color_map,
        "total": total,
        "month": "All",
    }



@app.post("/upload")
def upload_pdf():
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

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

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

    # Ensure Amount is numeric (so Jinja's '{:,.2f}' works)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

    # Save CSV
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
        summary=summary,
        preview=preview
    )

@app.get("/predictions/<path:filename>")
def download_prediction(filename):
    return send_from_directory(PRED_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
