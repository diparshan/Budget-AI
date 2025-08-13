import pdfplumber
import re
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def parse_walmart_pdf(pdf_path):
    transactions = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            matches = re.findall(r"([A-Za-z]{3} \d{1,2})\s+([A-Za-z]{3} \d{1,2})\s+(.+?)\s+\$(-?\d+\.\d{2})", text)
            for trans_date, post_date, description, amount in matches:
                transactions.append({
                    "Trans Date": trans_date,
                    "Post Date": post_date,
                    "Description": description.strip(),
                    "Category": "Unknown",  # Placeholder for unknown category
                    "Amount": float(amount)
                })
    return transactions

def predict_category_for_walmart_dl(pdf_path, tokenizer_path, label_encoder_path, model_path, max_len=50):
    # Parse the PDF to get transactions
    walmart_transactions = parse_walmart_pdf(pdf_path)
    walmart_df = pd.DataFrame(walmart_transactions)

    if walmart_df.empty:
        print("No transactions found in the PDF.")
        return pd.DataFrame()

    # Load the Tokenizer and Label Encoder
    tokenizer = joblib.load(tokenizer_path)
    label_encoder = joblib.load(label_encoder_path)

    # Preprocess descriptions (same as training data)
    walmart_df["Description_processed"] = walmart_df["Description"].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower().strip()))

    # Convert text to sequences and pad
    sequences = tokenizer.texts_to_sequences(walmart_df["Description_processed"])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    # Load the trained deep learning model
    model = load_model(model_path)

    # Predict categories
    predicted_probs = model.predict(padded_sequences)
    predicted_labels = np.argmax(predicted_probs, axis=1)

    # Map predicted labels back to original category names
    # Replace 'Unknown' category with predicted
    walmart_df["Category"] = label_encoder.inverse_transform(predicted_labels)
    walmart_df["User_Corrected_Category"] = "" # New column for user corrections

    #dropping description_processed column
    walmart_df.drop(columns=["Description_processed"], inplace=True)


    return walmart_df

if __name__ == "__main__":
    # Check if a PDF file was provided
    if len(sys.argv) < 2:
        print("Usage: python predict_category.py <pdf_file.pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]  # Take the first argument from terminal
    tokenizer_file = "./preprocessors/tokenizer.pkl"
    label_enc_file = "./preprocessors/label_encoder_dl.pkl"
    model_file = "./models/bidirectional_lstm_model.keras"

    
    # Create predictions folder if it doesn't exist
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)

    predicted_transactions_df_dl = predict_category_for_walmart_dl(
        pdf_file, tokenizer_file, label_enc_file, model_file
    )

    if not predicted_transactions_df_dl.empty:
        print("\nPredicted Categories for Walmart Transactions (Deep Learning Model):")
        print(predicted_transactions_df_dl[["Description", "Amount", "Category"]].to_string())

        # Create CSV path inside predictions folder
        output_csv_path = os.path.join(
            predictions_dir,
            f"{os.path.splitext(os.path.basename(pdf_file))[0]}_predictions.csv"
        )

        predicted_transactions_df_dl.to_csv(output_csv_path, index=False)
        print(f"\nPredicted transactions saved to {output_csv_path}")


