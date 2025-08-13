import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Map all near-duplicate category spellings/casing to a single canonical form
CANON_CATEGORIES = {
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

def _clean_description(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)                 # collapse whitespace
    s = re.sub(r"[^a-z0-9\s]", "", s)          # keep alnum + space
    return s

def _normalize_category(c: str):
    if pd.isna(c):
        return np.nan
    c_norm = re.sub(r"\s+", " ", str(c).strip()).lower()
    return CANON_CATEGORIES.get(c_norm, c_norm.title())

def preprocess_dl_data(df: pd.DataFrame, max_words: int = 5000, max_len: int = 50):
    df = df.copy()

    # Normalize/clean
    df["Category"] = df["Category"].apply(_normalize_category)
    df["Description"] = df["Description"].apply(_clean_description)

    # Drop rows missing essentials
    df = df[(df["Category"].notna()) & (df["Description"].str.len() > 0)]
    if df.empty:
        raise ValueError("No usable rows after cleaning. Check your input data.")

    # Tokenize descriptions
    tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
    tokenizer.fit_on_texts(df["Description"])
    sequences = tokenizer.texts_to_sequences(df["Description"])
    X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Category"])

    return X, y, tokenizer, label_encoder

if __name__ == "__main__":
    # Load combined data
    df = pd.read_csv("./data/combined_transactions_copy.csv")

    # Preprocess
    X_dl, y_dl, tokenizer, label_encoder = preprocess_dl_data(df)

    # Ensure output dir exists
    os.makedirs("./preprocessors", exist_ok=True)

    # Save artifacts
    joblib.dump(tokenizer, "./preprocessors/tokenizer.pkl")
    joblib.dump(label_encoder, "./preprocessors/label_encoder_dl.pkl")
    np.save("./preprocessors/X.npy", X_dl)
    np.save("./preprocessors/y.npy", y_dl)

    # Small summary for sanity-check
    print("Deep learning preprocessing complete.")
    print(f"- X shape: {X_dl.shape} | y shape: {y_dl.shape}")
    print(f"- Vocabulary size (capped): {min(len(tokenizer.word_index)+1, 5000)}")
    print(f"- Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
