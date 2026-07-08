from tensorflow import keras
from tensorflow.keras import layers

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def build_model(
    vocab_size: int,
    max_len: int,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    dropout: float = 0.3,
    num_labels: int = len(LABELS),
) -> keras.Model:
    """Bidirectional LSTM for multi-label toxic comment classification."""
    inputs = keras.Input(shape=(max_len,), name="tokens")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_labels, activation="sigmoid", name="labels")(x)

    model = keras.Model(inputs, outputs, name="jigsaw_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc", multi_label=True), "binary_accuracy"],
    )
    return model
