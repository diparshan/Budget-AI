# train_model.py

import os
# Quiet TF's CPU feature banner BEFORE importing tensorflow/keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

import random
import numpy as np
import joblib

import tensorflow as tf
tf.random.set_seed(42)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


# --------------------------------------------------------------------------------------
# Helper 1: Stratified split that ensures each class with >=2 samples appears in TEST
# --------------------------------------------------------------------------------------
def split_with_all_classes(X, y, test_size=0.2, random_state=42):
    """
    Deterministic, per-class split:
    - If a class has >=2 samples: put at least 1 in test and 1 in train.
    - If a class has <2 samples: keep all in train (can't be in both).
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    train_idx, test_idx = [], []

    unique_labels = np.unique(y)
    for cls in unique_labels:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        if n >= 2:
            # choose test count (at least 1, leave at least 1 for train)
            n_test = max(1, int(round(n * test_size)))
            n_test = min(n - 1, n_test)
            test_i = cls_idx[:n_test]
            train_i = cls_idx[n_test:]
            test_idx.extend(test_i.tolist())
            train_idx.extend(train_i.tolist())
        else:
            # too rare to split — keep in train only
            train_idx.extend(cls_idx.tolist())

    # Shuffle final indices for randomness
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------------------
# Helper 2 (optional): simple oversampling to the max class count
# --------------------------------------------------------------------------------------
def oversample_to_max(X, y, random_state=42):
    """
    Randomly oversample each class up to the size of the largest class.
    Returns new (X_res, y_res) shuffled.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()

    X_out, y_out = [], []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        n = len(cls_idx)
        if n == 0:
            continue
        # number to add
        n_add = target - n
        if n_add > 0:
            add_idx = rng.choice(cls_idx, size=n_add, replace=True)
            idx_all = np.concatenate([cls_idx, add_idx])
        else:
            idx_all = cls_idx
        X_out.append(X[idx_all])
        y_out.append(y[idx_all])

    X_res = np.concatenate(X_out, axis=0)
    y_res = np.concatenate(y_out, axis=0)

    # final shuffle
    perm = rng.permutation(len(y_res))
    return X_res[perm], y_res[perm]


def train_dl_model():
    # --------------------
    # Load preprocessed data & encoders
    # --------------------
    np.random.seed(42)
    random.seed(42)

    X_dl = np.load("./preprocessors/X.npy")               # shape: (N, max_len)
    y_dl = np.load("./preprocessors/y.npy")               # shape: (N,)
    tokenizer = joblib.load("./preprocessors/tokenizer.pkl")
    label_encoder = joblib.load("./preprocessors/label_encoder_dl.pkl")

    num_classes = len(label_encoder.classes_)
    max_len = X_dl.shape[1]

    # Effective vocab size (respect Tokenizer.num_words cap if set)
    num_words = getattr(tokenizer, "num_words", None)
    vocab_size = (len(tokenizer.word_index) + 1) if not num_words else min(num_words, len(tokenizer.word_index) + 1)

    print("Loaded data:")
    print(f"- X shape: {X_dl.shape} | y shape: {y_dl.shape}")
    print(f"- Vocab size (effective): {vocab_size}")
    print(f"- Classes ({num_classes}): {list(label_encoder.classes_)}")

    # --------------------
    # Train/Test split with class guarantee
    # --------------------
    X_train, X_test, y_train_int, y_test_int = split_with_all_classes(
        X_dl, y_dl, test_size=0.2, random_state=42
    )

    # Optional: oversample training set (toggle as needed)
    USE_OVERSAMPLING = False  # set True if you want to upsample minority classes
    if USE_OVERSAMPLING:
        X_train, y_train_int = oversample_to_max(X_train, y_train_int, random_state=42)

    # One-hot encode AFTER the split
    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_test = to_categorical(y_test_int, num_classes=num_classes)

    print("\nSplit summary:")
    print(f"- Train: {X_train.shape[0]} samples")
    print(f"- Test : {X_test.shape[0]} samples")

    # --------------------
    # Class weights (only if not oversampling)
    # --------------------
    class_weights_dict = None
    if not USE_OVERSAMPLING:
        # compute weights using *train* labels
        cw = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=y_train_int
        )
        class_weights_dict = {i: float(w) for i, w in enumerate(cw)}
        print("\nClass weights:", class_weights_dict)

    # --------------------
    # Model
    # --------------------
    embedding_dim = 100

    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(vocab_size, embedding_dim, mask_zero=True),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --------------------
    # Training
    # --------------------
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-5)
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )

    # --------------------
    # Evaluation
    # --------------------
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    all_labels = np.arange(num_classes)

    print("\n--- Deep Learning Model Classification Report (Optimized) ---")
    print(classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=list(label_encoder.classes_),
        zero_division=0,
        digits=4
    ))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("Label order:", list(label_encoder.classes_))

    # --------------------
    # Save
    # --------------------
    save_path = "./models/bidirectional_lstm_model.keras"
    model.save(save_path)
    print(f"\nBidirectional LSTM model saved as {save_path}")


if __name__ == "__main__":
    train_dl_model()
