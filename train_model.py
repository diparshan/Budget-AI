import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

def train_dl_model():
    # Load preprocessed data
    X_dl = np.load("./preprocessors/X.npy")
    y_dl = np.load("./preprocessors/y.npy")
    tokenizer = joblib.load("./preprocessors/tokenizer.pkl")
    label_encoder = joblib.load("./preprocessors/label_encoder_dl.pkl")

    num_classes = len(label_encoder.classes_)
    y_dl_categorical = to_categorical(y_dl, num_classes=num_classes)

    # Calculate class weights to handle imbalance
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_dl),
        y=y_dl
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_dl, y_dl_categorical, test_size=0.2, random_state=42, stratify=y_dl)

    # Model Parameters
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    max_len = X_dl.shape[1] # Length of padded sequences

    # Build Bidirectional LSTM Model
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64)),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Early Stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with class weights
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.1, 
        callbacks=[early_stopping], 
        class_weight=class_weights_dict)

    # Evaluate the model
    loss, accuracy = model.evaluate(
        X_test, 
        y_test, 
        verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Get predictions for classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n--- Deep Learning Model Classification Report (Optimized) ---")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Save the trained model
    model.save("./models/bidirectional_lstm_model.keras")
    print("Bidirectional LSTM model saved as bidirectional_lstm_model.keras")

if __name__ == "__main__":
    train_dl_model()


