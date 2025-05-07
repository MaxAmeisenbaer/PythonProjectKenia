import os
import re
import pandas as pd
from datetime import datetime



from lstm_model import create_model, train_model
from data_prepro import create_final_ds
from benchmark_szenario_sha import get_benchmark_config
from evaluate_model import calculate_all_metrics

def save_model_metadata(model_name, params, output_path="model_log.xlsx"):
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        **params
    }

    if os.path.exists(output_path):
        df = pd.read_excel(output_path)
        df = pd.concat([df, pd.DataFrame([metadata])], ignore_index=True)
    else:
        df = pd.DataFrame([metadata])

    df.to_excel(output_path, index=False)

def generate_model_name(config_name, target_feature, output_path="model_log.xlsx"):
    station, target = target_feature.split("_")
    prefix = f"LSTM_{station}_{config_name}_{target}"

    existing_numbers = []
    if os.path.exists(output_path):
        import pandas as pd
        df = pd.read_excel(output_path)
        pattern = rf"{re.escape(prefix)}_(\d+)"
        for name in df["model_name"]:
            match = re.match(pattern, name)
            if match:
                existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1
    return f"{prefix}_{next_number:03d}"

def main():
    stations, measurements, target_feature, config_name = get_benchmark_config()

    model_config = {
        "nodes_lstm": 20,
        "nodes_dense": None,
        "dropout": 0.1,
        "metric": "mse",
        "learning_rate": 0.001,
        "batch_size": 32,
        "seq_length": 2,
        "epochs": 50
    }

    train_ds, val_ds, test_ds, train_df, test_df, val_df = create_final_ds(
        station="SHA",
        stations=stations,
        target_feature=target_feature,
        batch_size=model_config["batch_size"],
        seq_length=model_config["seq_length"],
        measurements=measurements
    )

    model, early_stopping = create_model(
        nodes_lstm=model_config["nodes_lstm"],
        nodes_dense=model_config["nodes_dense"],
        dropout=model_config["dropout"],
        metric=model_config["metric"],
        learning_rate=model_config["learning_rate"]
    )

    history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        early_stopping=early_stopping,
        metric=model_config["metric"],
        epochs=model_config["epochs"]
    )
    return model, test_ds, model_config, target_feature, config_name

if __name__ == "__main__":
    model, test_ds, model_config, target_feature, config_name = main()

    metrics_result = calculate_all_metrics(model, test_ds)

    model_name = generate_model_name(config_name, target_feature)

    save_model_metadata(
        model_name=model_name,
        params={**model_config, **metrics_result}
    )
