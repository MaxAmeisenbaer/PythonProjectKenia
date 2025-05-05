import tensorflow as tf
import numpy as np
import matplotlib as plt

class NSEMetric(tf.keras.metrics.Metric):
    def __init__(self, name="nse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.var = self.add_weight(name="var", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1,))
        self.sse.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.var.assign_add(tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return 1.0 - (self.sse / (self.var + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.sse.assign(0.0)
        self.var.assign(0.0)
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
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(nodes_lstm, return_sequences=True),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.GlobalMaxPooling1D(),

        tf.keras.layers.Dense(1, activation="linear")
    ])

    metrics = {"root_mean_squared_error": tf.keras.metrics.RootMeanSquaredError(),
               "r_square": tf.keras.metrics.R2Score(dtype=tf.float32),
               "mean_absolute_error": tf.keras.metrics.MeanAbsoluteError(),
               "nse": NSEMetric(),
               }

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[metrics[metric]])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_root_mean_squared_error",
        patience=20,
        min_delta=0.001,
    )

    return model, early_stopping


def train_model(model, train_ds, val_ds, early_stopping, metric, EPOCHS):
    kge_callback = KGECallback(val_ds)
    history = model.fit(train_ds, epochs=EPOCHS,
                        validation_data=val_ds,
                        callbacks=[early_stopping, kge_callback])

    model.save()

    # list all data in history
    print(history.history.keys())
    # visualize history for accuracy
    plt.plot(history.history[f'{metric}'])
    plt.plot(history.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel(f'{metric}')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # visualize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()