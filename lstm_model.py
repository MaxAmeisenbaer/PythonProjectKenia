import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Sets the Python, NumPy, and PyTorch random number generators for reproducible runs.

    :param seed: Seed for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN kernels (relevant when training on GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_nse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Computes the Nash-Sutcliffe efficiency (NSE).

    NSE = 1 - (SSE / Var(y_true))
    - Values close to 1: high model performance
    - Values <= 0: the mean would be better than the model

    :param y_true: Observed values
    :param y_pred: Predicted values

    :return: Average NSE value over all target features as float
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
    Computes the Mean Bias Error (MBE).

    - MBE > 0: overestimation
    - MBE < 0: underestimation
    - MBE = 0: perfect prediction

    :param y_true: Observed values
    :param y_pred: Predicted values

    :return: MBE value as float
    """
    return (y_pred - y_true).mean().item()


def _kling_gupta_efficiency(sim: np.ndarray, obs: np.ndarray,
                           mask: np.ndarray | None = None) -> float:
    """
    Computes the Kling-Gupta efficiency (KGE).

    :param sim:  Model predictions [n_samples, n_targets]
    :param obs:  Observations [n_samples, n_targets]
    :param mask: Optional observation mask [n_samples, n_targets]
                 (1 = observed, 0 = missing/imputed)

    :return: Average KGE value (maximum 1, the closer to 1 the better)
    """
    n_targets = sim.shape[1]
    kge_values = []

    if mask is not None:
        mask = np.asarray(mask)

    for i in range(n_targets):
        s = sim[:, i]
        o = obs[:, i]

        if mask is not None:
            valid = mask[:, i] > 0
            s = s[valid]
            o = o[valid]

        finite = np.isfinite(s) & np.isfinite(o)
        s = s[finite]
        o = o[finite]

        if s.size < 2:
            kge_values.append(np.nan)
            continue

        std_s = np.std(s)
        std_o = np.std(o)
        mean_o = np.mean(o)
        if std_s <= 1e-12 or std_o <= 1e-12 or abs(mean_o) <= 1e-12:
            kge_values.append(np.nan)
            continue

        r = np.corrcoef(s, o)[0, 1]
        if not np.isfinite(r):
            kge_values.append(np.nan)
            continue

        alpha = std_s / std_o
        beta  = np.mean(s) / mean_o
        kge_values.append(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

    valid_kge = [value for value in kge_values if np.isfinite(value)]
    return float(np.mean(valid_kge)) if valid_kge else float("nan")


class LSTMModel(nn.Module):
    """
    LSTM model with an optional dense intermediate layer.

    :param n_features:   Number of input features
    :param n_targets:    Number of target variables
    :param nodes_lstm:   Number of neurons in the LSTM layer
    :param nodes_dense:  Number of neurons in the dense layer (0 = none)
    :param dropout:      Dropout rate
    :param num_layers:  Number of stacked LSTM layers
    """
    def __init__(self, n_features: int, n_targets: int, nodes_lstm: int,
                 nodes_dense: int, dropout: float, num_layers: int = 2):
        super().__init__()

        self.lstm    = nn.LSTM(
            input_size=n_features,
            hidden_size=nodes_lstm,
            num_layers=num_layers,
            batch_first=True,          # ← Input shape: [batch, seq_len, features]
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
        Forward pass through the model.

        :param x: Input tensor [batch_size, seq_length, n_features]

        :return:  Output tensor [batch_size, 1]
        """
        out, _ = self.lstm(x)  # [batch, seq_len, nodes_lstm]
        out = out[:, -1, :]  # [batch, nodes_lstm] ← last time step
        out = self.dropout(out)

        if self.dense is not None:
            out = self.relu(self.dense(out))     # [batch, nodes_dense]

        return self.output_layer(out)            # [batch, n_targets]


def _masked_mse_loss(y_pred, y_true, y_mask):
    """
    Computes the MSE only for values where the mask is 1.

    :param y_pred: Prediction [batch, n_targets]
    :param y_true: True values [batch, n_targets]
    :param y_mask: Mask (1=real, 0=interpolated) [batch, n_targets]

    :return: Scalar loss
    """
    # Square the element-wise difference
    squared_error = (y_pred - y_true) ** 2
    # Multiply by the mask
    masked_error = squared_error * y_mask
    # Compute the average over the real values (sum / number of real values) - epsilon so that it does not divide by 0
    loss = torch.sum(masked_error) / (torch.sum(y_mask) + 1e-8)

    return loss


def _masked_mse_euclidean_loss(y_pred, y_true, y_mask, eps: float = 1e-8):
    """
    Computes a per-target Euclidean variant of the MSE.

    For each target, the MSE is first computed, then its square root (RMSE)
    is taken, and finally the mean over all observed targets is squared.
    As a result, each target contributes equally to the loss regardless of
    its number of observations, while the loss remains in MSE-like units.

        L = (mean_t sqrt(MSE_t))^2

    :param y_pred: Prediction [batch, n_targets]
    :param y_true: True values [batch, n_targets]
    :param y_mask: Mask (1=real, 0=interpolated) [batch, n_targets]
    :param eps:    Small value for numerical stability

    :return:       Scalar loss
    """
    squared_error = (y_pred - y_true) ** 2 * y_mask

    # Compute MSE separately per target
    per_target_sq = torch.sum(squared_error, dim=0)
    per_target_count = torch.sum(y_mask, dim=0)

    # Exclude targets without observed values
    valid = per_target_count > 0
    if not torch.any(valid).item():
        # Keep connected to y_pred so that backward() also works for a
        # fully masked batch.
        return torch.sum(y_pred) * 0.0

    per_target_mse = per_target_sq[valid] / (per_target_count[valid] + eps)
    per_target_rmse = torch.sqrt(per_target_mse + eps)

    # Square the mean of the RMSEs -> MSE-like units
    return torch.mean(per_target_rmse) ** 2


def create_model(n_features: int, n_targets: int, nodes_lstm: int, nodes_dense: int,
                 dropout: float, learning_rate: float, num_layers: int = 2,
                 euclidean: bool = False):
    """
    Creates an LSTM model and the optimizer.

    :param n_features:    Number of input features
    :param n_targets:     Number of target features
    :param nodes_lstm:    Neurons in the LSTM layer
    :param nodes_dense:   Neurons in the dense layer (0 = none)
    :param dropout:       Dropout rate
    :param learning_rate: Learning rate of the Adam optimizer
    :param num_layers:    Number of layers (model depth), default=2
    :param euclidean:     If True, the MSE is aggregated per target in a
                          Euclidean way, (mean_t sqrt(MSE_t))^2, instead of
                          averaging over all values.

    :return: model, optimizer, loss_fn
    """
    model     = LSTMModel(n_features, n_targets, nodes_lstm, nodes_dense, dropout, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn   = _masked_mse_euclidean_loss if euclidean else _masked_mse_loss

    return model, optimizer, loss_fn


def train_model(model: nn.Module, train_loader, val_loader,
                optimizer, loss_fn, epochs: int, patience: int = 5):
    """
    Trains the model with a manual training loop and early stopping.

    :param model:        The LSTM model
    :param train_loader: DataLoader for training data
    :param val_loader:   DataLoader for validation data
    :param optimizer:    Adam optimizer
    :param loss_fn:      Loss function (MSE or per-target Euclidean MSE)
    :param epochs:       Maximum number of epochs
    :param patience:     Patience for early stopping

    :return: history (dict with train_loss and val_loss per epoch)
    """
    #Commented out - scheduler can be implemented - also raise early stopping
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
        for x_batch, y_batch, y_mask_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss   = loss_fn(y_pred, y_batch, y_mask_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_losses = []
        all_y_true = []
        all_y_pred = []
        all_y_mask = []
        with torch.no_grad():
            for x_batch, y_batch, y_mask_batch in val_loader:
                y_pred = model(x_batch)
                loss   = loss_fn(y_pred, y_batch, y_mask_batch)
                val_losses.append(loss.item())

                all_y_true.append(y_batch.cpu().numpy()) #Change due to multi-target
                all_y_pred.append(y_pred.cpu().numpy()) #Change due to multi-target
                all_y_mask.append(y_mask_batch.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        all_y_true_arr = np.concatenate(all_y_true, axis=0)
        all_y_pred_arr = np.concatenate(all_y_pred, axis=0)
        all_y_mask_arr = np.concatenate(all_y_mask, axis=0)
        val_kge = _kling_gupta_efficiency(all_y_pred_arr, all_y_true_arr, all_y_mask_arr)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_kge"].append(val_kge)

        #Output in the TUI (Verbose)
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

    # Restore the best weights
    model.load_state_dict(best_weights)
    return history
