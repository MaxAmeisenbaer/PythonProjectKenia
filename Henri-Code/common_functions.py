import tensorflow as tf
import tensorflow_addons as tfa


def create_model(nodes_lstm, nodes_dense, dropout, metric, learning_rate):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(nodes_lstm, return_sequences=True),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.GlobalMaxPooling1D(),

        tf.keras.layers.Dense(1, activation="linear")
    ])

    metrics = {"root_mean_squared_error": tf.metrics.RootMeanSquaredError(),
               "r_square": tfa.metrics.RSquare(dtype=tf.float32),
               "mean_absolute_error": tf.metrics.MeanAbsoluteError(),
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
