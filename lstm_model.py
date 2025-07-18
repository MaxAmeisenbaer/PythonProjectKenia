import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class NSEMetric(tf.keras.metrics.Metric):
    def __init__(self, name="nse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.y_true_sum = self.add_weight(name="y_true_sum", initializer="zeros")
        self.y_true_sq_sum = self.add_weight(name="y_true_sq_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1,))

        self.sse.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.y_true_sum.assign_add(tf.reduce_sum(y_true))
        self.y_true_sq_sum.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        mean_y_true = self.y_true_sum / self.count
        total_var = self.y_true_sq_sum - self.count * tf.square(mean_y_true)
        return 1.0 - (self.sse / (total_var + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.sse.assign(0.0)
        self.y_true_sum.assign(0.0)
        self.y_true_sq_sum.assign(0.0)
        self.count.assign(0.0)


class MBEMetric(tf.keras.metrics.Metric):
    def __init__(self, name="mean_bias_error", **kwargs):
        super(MBEMetric, self).__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name="total_error", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = y_pred - y_true
        if sample_weight is not None:
            error = tf.multiply(error, sample_weight)
        self.total_error.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total_error / self.count

    def reset_states(self):
        self.total_error.assign(0.0)
        self.count.assign(0.0)

def kling_gupta_efficiency(sim, obs):
    sim = np.array(sim)
    obs = np.array(obs)
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

class KGECallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []

        for x_batch, y_batch in self.val_ds:
            preds = self.model.predict(x_batch, verbose=0)
            y_true.extend(y_batch.numpy().flatten())
            y_pred.extend(preds.flatten())

        kge = kling_gupta_efficiency(y_pred, y_true)
        print(f"Epoch {epoch + 1}: KGE = {kge:.4f}")

def create_model(nodes_lstm, nodes_dense, dropout, metric, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(nodes_lstm, return_sequences=True))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.GlobalMaxPooling1D())

    if nodes_dense > 0:
        model.add(tf.keras.layers.Dense(nodes_dense, activation="relu"))

    model.add(tf.keras.layers.Dense(1, activation="linear"))

    metrics = {"root_mean_squared_error": tf.keras.metrics.RootMeanSquaredError(),
               "mean_squared_error": tf.keras.metrics.MeanSquaredError(),
               "mean_absolute_error": tf.keras.metrics.MeanAbsoluteError(),
               "r_square": tf.keras.metrics.R2Score(dtype=tf.float32),
               "nse": NSEMetric(),
               "mbe": MBEMetric(),
               }

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[metrics[metric]])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_squared_error",
        patience=5,
        min_delta=0.0,
        restore_best_weights=True
    )

    return model, early_stopping


def train_model(model, train_ds, val_ds, early_stopping,
                metric, epochs,
                test_ds=None, full_ds=None, timestamps_full=None,
                model_name=None, output_dir=None, x_full=None, scaler_y=None):
    """
    Trainiert das Modell mit optionaler Evaluation und vollständiger Speicherung.

    Args:
        model: Das zu trainierende Modell.
        train_ds, val_ds: Trainings- und Validierungsdaten.
        early_stopping: Keras EarlyStopping Callback.
        metric: Bewertungsmetrik.
        epochs: Anzahl Epochen.
        test_ds: Optionaler Testdatensatz für Metriken.
        full_ds: Optionaler vollständiger Datensatz für Speicherung.
        timestamps_full, model_name, output_dir, X_full, scaler_y: Optional – nur für vollständige Speicherung nötig.
    """

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=0
    )


    # Optional: vollständige Analyse-Dateien speichern
    if full_ds and timestamps_full is not None and x_full is not None and scaler_y is not None:
        from evaluate_model import evaluate_and_store_full_predictions
        evaluate_and_store_full_predictions(
            model=model,
            full_ds=full_ds,
            timestamps=timestamps_full,
            output_dir=output_dir,
            x_full=x_full,
            scaler_y=scaler_y
        )
        return history,
