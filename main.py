import sys
import os
import pandas as pd
from predict_category import predict_category_for_walmart_dl

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_file1.pdf> [pdf_file2.pdf ...]")
        sys.exit(1)

    pdf_files = sys.argv[1:]

    # Define paths to models and preprocessors relative to main.py
    models_dir = "models"
    preprocessors_dir = "preprocessors"

    # Deep Learning Model paths
    dl_tokenizer_file = os.path.join(preprocessors_dir, "tokenizer.pkl")
    dl_label_enc_file = os.path.join(preprocessors_dir, "label_encoder_dl.pkl")
    dl_model_file = os.path.join(models_dir, "bidirectional_lstm_model.keras")

    # Create predictions folder
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)

    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file}...")

        # Run Deep Learning Model Prediction
        try:
            dl_predicted_df = predict_category_for_walmart_dl(
                pdf_file, dl_tokenizer_file, dl_label_enc_file, dl_model_file
            )
            if not dl_predicted_df.empty:
                dl_predicted_df = dl_predicted_df[["Trans Date", "Post Date", "Description", "Category", "Amount"]]

                # Save inside predictions/ with same naming pattern
                base_name = os.path.splitext(os.path.basename(pdf_file))[0]
                output_dl_csv = os.path.join(predictions_dir, f"{base_name}_predictions.csv")

                dl_predicted_df.to_csv(output_dl_csv, index=False)
                print(f"Deep Learning predictions saved to {output_dl_csv}")
            else:
                print(f"No transactions predicted by Deep Learning for {pdf_file}.")
        except Exception as e:
            print(f"Error running Deep Learning prediction for {pdf_file}: {e}")

if __name__ == "__main__":
    main()
