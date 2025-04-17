import tensorflow as tf


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
