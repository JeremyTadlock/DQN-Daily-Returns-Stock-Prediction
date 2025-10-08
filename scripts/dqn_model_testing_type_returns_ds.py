# Imports
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Config (match training!)
model_path = "trained_dqn_predict_daily_return_3yrs_2.pth"

normal_res_student = "trained_dqn_predict_daily_return_3yrs_2.pth"
higher_res_student = "trained_dqn_predict_daily_return_3yrs_2_finer_res.pth"

if model_path == normal_res_student:
    print("testing normal res student")
    ACTION_BINS = np.arange(-4.0, 4.0 + 0.5, 0.5)  # MUST match training
elif model_path == higher_res_student:
    print("testing higher res student")
    ACTION_BINS = np.linspace(-2.5, 2.5, 51)  # using linspace for -2.5..2.5 @ 0.1
else:
    print("error, model path is not on the selected list. assuming normal res")
    ACTION_BINS = np.arange(-4.0, 4.0 + 0.5, 0.5)

WINDOW_SIZE = 25

# DQN Network (same as training, but optimizer unused)
class DQN(nn.Module):
    def __init__(self, state_size, action_size, learning_rate=0.001, device=None, verbose=False, is_target=False):
        super(DQN, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.is_target = is_target

        # Architecture (must match training)
        self.fc1 = nn.Linear(state_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.value_fc = nn.Linear(256, 256)
        self.value = nn.Linear(256, 1)
        self.advantage_fc = nn.Linear(256, 256)
        self.advantage = nn.Linear(256, action_size)

        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        value = torch.relu(self.value_fc(x))
        value = self.value(value)
        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Testing Data Prep
def get_testing_state(ticker, window_size=WINDOW_SIZE, period="60d"):
    """
    Returns:
      state_norm: np.ndarray of shape (window_size,)
      predicted_date: next business day after the last complete bar
      data: full DataFrame for inspection
    """
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data is None or data.empty:
        raise ValueError(f"No data downloaded for ticker: {ticker}")

    data["DailyReturn"] = data["Close"].pct_change() * 100.0
    data.dropna(inplace=True)

    if len(data) < window_size:
        raise ValueError(f"Not enough data to form a {window_size}-day window; got {len(data)} rows.")

    # last complete day
    last_complete_date = data.index[-1]
    predicted_date = last_complete_date + pd.offsets.BDay(1)

    # raw returns window (no normalization here; we’ll normalize with train stats)
    returns = data["DailyReturn"].iloc[-window_size:].values.astype(np.float32)

    return returns, predicted_date, data

# Load model + training normalization  (ROBUST VERSION)
def load_model_and_stats(model_path, state_size, device=None):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # safer load; fall back if your torch doesn't support weights_only
    try:
        ckpt = torch.load(model_path, map_location=dev, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=dev)

    # Support both raw state_dict or checkpoint dict
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # infer action size from checkpoint
    if "advantage.bias" in state_dict:
        expected_actions = state_dict["advantage.bias"].shape[0]
    else:
        keys = [k for k in state_dict if "advantage" in k and ("weight" in k or "bias" in k)]
        if not keys:
            raise ValueError("Could not infer action size from checkpoint (no advantage layer found).")
        # prefer bias if present
        bkeys = [k for k in keys if k.endswith("bias")]
        if bkeys:
            expected_actions = state_dict[bkeys[0]].shape[0]
        else:
            # weight shape [out, in]
            expected_actions = state_dict[keys[0]].shape[0]

    # sanity check bins
    if expected_actions != len(ACTION_BINS):
        raise ValueError(
            f"ACTION_BINS length ({len(ACTION_BINS)}) does not match checkpoint action size ({expected_actions}). "
            f"Switch ACTION_BINS to the set that matches this model."
        )

    # build DQN with the correct output width
    dqn = DQN(state_size=state_size, action_size=expected_actions, device=dev)
    dqn.load_state_dict(state_dict, strict=True)
    dqn.eval()

    # pull training stats if present
    if isinstance(ckpt, dict):
        train_mean = float(ckpt.get("train_mean", np.nan))
        train_std  = float(ckpt.get("train_std",  np.nan))
    else:
        train_mean, train_std = np.nan, np.nan

    return dqn, train_mean, train_std

# Predict next-day return
def predict_next_return(ticker="SPY", model_path="trained_dqn_predict_daily_return_3yrs.pth",
                        window_size=WINDOW_SIZE, period="60d"):
    # Load model and train stats (action size inferred from checkpoint)
    dqn, train_mean, train_std = load_model_and_stats(model_path, state_size=window_size)

    # Get last window of returns (unnormalized)
    returns_window, predicted_date, data = get_testing_state(ticker, window_size=window_size, period=period)

    # Normalize with TRAINING stats if present; else fallback to window stats
    if not (np.isnan(train_mean) or np.isnan(train_std)):
        mean = train_mean
        std = train_std if train_std > 0 else 1e-8
        norm_source = "training stats"
    else:
        mean = returns_window.mean()
        std = returns_window.std() + 1e-8
        norm_source = "window stats (fallback)"

    state_norm = (returns_window - mean) / std

    # Forward pass
    state_tensor = torch.from_numpy(state_norm).float().to(dqn.device).unsqueeze(0)
    with torch.no_grad():
        q_values = dqn(state_tensor)
        action_idx = torch.argmax(q_values, dim=1).item()
        predicted_return = ACTION_BINS[action_idx]

    return {
        "predicted_date": predicted_date,
        "predicted_return_pct": float(predicted_return),
        "norm_source": norm_source,
        "last_window_returns_pct": returns_window,
        "raw_df": data
    }


# Run
if __name__ == "__main__":
    ticker = "SPY"

    out = predict_next_return(ticker=ticker, model_path=model_path, window_size=WINDOW_SIZE, period="60d")

    # Print a compact summary
    print("\nNormalization source:", out["norm_source"])
    print("Prediction will be made for:", out["predicted_date"].strftime("%Y-%m-%d"))
    print("Predicted return:", f"{out['predicted_return_pct']:.2f}%")

    # Optional: show the last 25 actual returns
    print("\nOriginal returns (last 25 days, %):")
    print(np.array2string(out["last_window_returns_pct"], precision=4, floatmode="fixed"))
