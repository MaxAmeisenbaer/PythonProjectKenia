# Model

**All models with interval=6 (1hour)**
- 
    LSTM(64)
    Dropout(0.2)
    LSTM(128)
    Dropout(0.2)
    GlobalMaxPooling1D()
    Dense(64, activation="relu")

    Dense(1, activation="linear")

- loss: MeanSquaredError
- optimizer: Adam (learning rate: 0.001)
- batch_size: 32
- sequence_length: 24


**All models with interval 18 (3hour)**
-
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.2)
    Dense(1, activation="linear")
- batch size: 32
- seq_length: 8 (24h)

- R2: 0.8299 - val_R2: 0.8645 - test_R2: 0.0 
- rmse: 0.085 - val_rmse:0.07 - train_rmse: 0.13
- mae: 0.0614 - val_mae: 0.0511 - test_mae: 0.1