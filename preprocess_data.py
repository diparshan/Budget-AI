import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import joblib
import numpy as np

def preprocess_dl_data(df, max_words=5000, max_len=50):
    # Clean and normalize transaction descriptions
    df["Description"] = df["Description"].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower().strip()))

    # Tokenization
    tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
    tokenizer.fit_on_texts(df["Description"])
    sequences = tokenizer.texts_to_sequences(df["Description"])

    # Padding sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # Label Encoding for Categories
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Category"])

    return padded_sequences, y, tokenizer, label_encoder

if __name__ == "__main__":
    df = pd.read_csv("./data/combined_transactions.csv")
    X_dl, y_dl, tokenizer, label_encoder = preprocess_dl_data(df)

    # Save preprocessed data and encoders for later use
    joblib.dump(tokenizer, "./preprocessors/tokenizer.pkl")
    joblib.dump(label_encoder, "./preprocessors/label_encoder_dl.pkl")
    np.save("./preprocessors/X.npy", X_dl)
    np.save("./preprocessors/y.npy", y_dl)

    print("Deep learning data preprocessing complete. Tokenizer, Label Encoder, X_dl.npy, and y_dl.npy saved.")


