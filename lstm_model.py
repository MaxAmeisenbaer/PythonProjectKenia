import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def compute_nse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Berechnet die Nash-Sutcliffe-Effizienz (NSE).

    NSE = 1 - (SSE / Var(y_true))
    - Werte nahe 1: hohe Modellgüte
    - Werte <= 0: Mittelwert wäre besser als Modell

    :param y_true: Beobachtete Werte
    :param y_pred: Vorhergesagte Werte
    :return: Durchschnittlischer NSE-Wert über alle Target-Features als float
    """
    n_targets = y_true.shape[1]
    nse_values = []

    for i in range(n_targets):
        yt = y_true[:, i].reshape(-1)
        yp = y_pred[:, i].reshape(-1)
        sse = torch.sum((yt - yp) ** 2)
        var = torch.sum((yt - torch.mean(yt)) ** 2)
        nse_values.append(1.0 - (sse / (var + 1e-8)).item())

    return np.mean(nse_values)


def compute_mbe(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Berechnet den Mean Bias Error (MBE).

    - MBE > 0: Überschätzung
    - MBE < 0: Unterschätzung
    - MBE = 0: Perfekte Vorhersage

    :param y_true: Beobachtete Werte
    :param y_pred: Vorhergesagte Werte
    :return: MBE-Wert als float
    """
    return (y_pred - y_true).mean().item()


def kling_gupta_efficiency(sim: np.ndarray, obs: np.ndarray) -> float:
    """
    Berechnet die Kling-Gupta-Effizienz (KGE).

    :param sim: Modellvorhersagen
    :param obs: Beobachtungen
    :return: Durchschnittlicher KGE-Wert (maximal 1, je näher an 1 desto besser)
    """
    n_targets = sim.shape[1]
    kge_values = []

    for i in range(n_targets):
        s = sim[:, i]
        o = obs[:, i]
        r     = np.corrcoef(s, o)[0, 1]
        alpha = np.std(s) / np.std(o)
        beta  = np.mean(s) / np.mean(o)
        kge_values.append(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

    return np.mean(kge_values)


class LSTMModel(nn.Module):
    """
    LSTM-Modell mit optionaler Dense-Zwischenschicht.

    :param n_features:   Anzahl der Eingabe-Features
    :param n_targets:    Anzahl der Zielvariablen
    :param nodes_lstm:   Anzahl der Neuronen in der LSTM-Schicht
    :param nodes_dense:  Anzahl der Neuronen in der Dense-Schicht (0 = keine)
    :param dropout:      Dropout-Rate
    :param num_layers:  Anzahl gestapelter LSTM-Layer
    """
    def __init__(self, n_features: int, n_targets: int, nodes_lstm: int,
                 nodes_dense: int, dropout: float, num_layers: int = 2):
        super().__init__()

        self.lstm    = nn.LSTM(
            input_size=n_features,
            hidden_size=nodes_lstm,
            num_layers=num_layers,
            batch_first=True,          # ← Input-Shape: [batch, seq_len, features]
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

        if nodes_dense > 0:
            self.dense = nn.Linear(nodes_lstm, nodes_dense)
            self.relu  = nn.ReLU()
        else:
            self.dense = None

        self.output_layer = nn.Linear(
            nodes_dense if nodes_dense > 0 else nodes_lstm, n_targets
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das Modell.

        :param x: Eingabe-Tensor [batch_size, seq_length, n_features]
        :return:  Ausgabe-Tensor [batch_size, 1]
        """
        out, _ = self.lstm(x)  # [batch, seq_len, nodes_lstm]
        out = out[:, -1, :]  # [batch, nodes_lstm] ← letzter Zeitschritt
        out = self.dropout(out)

        if self.dense is not None:
            out = self.relu(self.dense(out))     # [batch, nodes_dense]

        return self.output_layer(out)            # [batch, n_targets]


def create_model(n_features: int, n_targets: int, nodes_lstm: int, nodes_dense: int,
                 dropout: float, learning_rate: float, num_layers: int = 2):
    """
    Erstellt ein LSTM-Modell und den Optimierer.

    :param n_features:    Anzahl der Eingabe-Features
    :param nodes_lstm:    Neuronen in der LSTM-Schicht
    :param nodes_dense:   Neuronen in der Dense-Schicht (0 = keine)
    :param dropout:       Dropout-Rate
    :param learning_rate: Lernrate des Adam-Optimierers
    :return: model, optimizer, loss_fn
    """
    model     = LSTMModel(n_features, n_targets, nodes_lstm, nodes_dense, dropout, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn   = nn.MSELoss()

    return model, optimizer, loss_fn


def train_model(model: nn.Module, train_loader, val_loader,
                optimizer, loss_fn, epochs: int, patience: int = 5):
    """
    Trainiert das Modell mit manuellem Training Loop und EarlyStopping.

    :param model:        Das LSTM-Modell
    :param train_loader: DataLoader für Trainingsdaten
    :param val_loader:   DataLoader für Validierungsdaten
    :param optimizer:    Adam-Optimierer
    :param loss_fn:      Verlustfunktion (MSE)
    :param epochs:       Maximale Epochenanzahl
    :param patience:     Geduld für EarlyStopping
    :return: history (dict mit train_loss und val_loss pro Epoche)
    """
    #Auskommentiert - Scheduler kann implementiert werden - earlyStopping mit hochstellen
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    #)

    history          = {"train_loss": [], "val_loss": [], "val_kge": []}
    best_val_loss    = float("inf")
    best_weights     = None
    patience_counter = 0

    for epoch in range(epochs):

        # --- Training ---
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss   = loss_fn(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validierung ---
        model.eval()
        val_losses = []
        all_y_true = []
        all_y_pred = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                loss   = loss_fn(y_pred, y_batch)
                val_losses.append(loss.item())

                all_y_true.append(y_batch.cpu().numpy()) #Änderung wgn MultiTarget
                all_y_pred.append(y_pred.cpu().numpy()) #Änderung wgn MultiTarget

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        all_y_true_arr = np.concatenate(all_y_true, axis=0)
        all_y_pred_arr = np.concatenate(all_y_pred, axis=0)
        val_kge = kling_gupta_efficiency(all_y_pred_arr, all_y_true_arr)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_kge"].append(val_kge)

        current_lr=optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} "
              f"| Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} " 
              f"| KGE: {val_kge:.4f}"
              f"| LR: {current_lr:.2e}")

        # --- EarlyStopping ---
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early Stopping nach Epoche {epoch+1}")
                break

    # Beste Gewichte wiederherstellen
    model.load_state_dict(best_weights)
    return history
