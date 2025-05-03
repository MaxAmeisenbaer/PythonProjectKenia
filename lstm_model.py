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


def train_model(metric, EPOCHS):
    history = model.fit(train_ds, epochs=EPOCHS,
                        validation_data=val_ds)

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