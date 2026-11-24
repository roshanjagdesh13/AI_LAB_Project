import itertools

import tensorflow as tf

from src.config import SEED

GRID = {"num_nodes": [16, 32], "dropout": [0, 0.2], "lr": [0.01, 0.005], "batch_size": [32, 64]}


def build_model(n_features, num_nodes, dropout, lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(num_nodes, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_nodes, activation="relu"),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation="sigmoid"),     # P(gamma)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


class _EpochReporter(tf.keras.callbacks.Callback):
    """Forwards the real per-epoch Keras logs to a plain function (used for live UI charts)."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def on_epoch_end(self, epoch, logs=None):
        self.fn(epoch + 1, {k: float(v) for k, v in (logs or {}).items()})


def train_neural_net(X_bal, y_bal, X_valid, y_valid, epochs=100, params=None,
                     on_combo=None, on_epoch=None):
    """
    Grid search over nodes / dropout / learning rate / batch size, or a single fit when
    `params` is given (quick mode). Early stopping and model selection both use the real
    (un-oversampled) validation set.
    on_combo(i, n, params) is called before each combination, on_epoch(epoch, logs) after each epoch.
    Returns (best_model, history_dict, info).
    """
    keys = list(GRID)
    combos = [params] if params else [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    best = {"val_loss": float("inf")}
    for i, p in enumerate(combos, 1):
        if on_combo:
            on_combo(i, len(combos), p)
        tf.keras.utils.set_random_seed(SEED)
        model = build_model(X_bal.shape[1], p["num_nodes"], p["dropout"], p["lr"])
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                      restore_best_weights=True)]
        if on_epoch:
            callbacks.append(_EpochReporter(on_epoch))
        hist = model.fit(X_bal, y_bal, epochs=epochs, batch_size=p["batch_size"],
                         validation_data=(X_valid, y_valid), verbose=0, callbacks=callbacks)
        val_loss = model.evaluate(X_valid, y_valid, verbose=0)[0]
        if val_loss < best["val_loss"]:
            best = {"val_loss": float(val_loss), "model": model, "params": dict(p),
                    "history": {k: [float(x) for x in v] for k, v in hist.history.items()}}
    info = {"best_params": best["params"], "val_loss": best["val_loss"],
            "epochs_trained": len(best["history"]["loss"])}
    return best["model"], best["history"], info
