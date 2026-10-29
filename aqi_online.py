"""
Online-learning, dual-view CNN forecaster for city-level hourly AQI (Delhi).

The data: one Excel file per month, rows = days, columns = 00:00 ... 23:00, values = AQI.
The month and year are read from the FILE NAME (e.g. ..._January_2026_Hourly.xlsx).

Two views of the same hourly series feed two convolutional branches:
    Short-term view : last 24 hours                -> 1D CNN  (recent trend)
    Daily view      : last 7 days as a 7 x 24 grid -> 2D CNN  (same-hour-yesterday patterns)
The branches are fused with the current AQI level and hour-of-day, and a dense head predicts
the CHANGE in AQI for each of the next 24 hours.

Commands
--------
  init      python aqi_online.py init --data_dir data --pattern "*City_Level*.xlsx"
            Builds the history from the given months and trains the base model.

  update    python aqi_online.py update --file AQI_Data_City_Level_Delhi_March_2026_Hourly.xlsx
            1) forecasts the new month with the CURRENT model (honest test-then-train),
            2) then trains the model on the new data (+ a replay sample of old data),
            3) saves everything and writes the next-24-hour forecast.

  forecast  python aqi_online.py forecast
            Writes the next 24 hours of forecast from the latest saved state.

  simulate  python aqi_online.py simulate --data_dir data --init_months 4 --variants dual,short,daily
            Replays your months as if they arrived one by one: trains on the first N months, then
            "updates" month by month. Compares Persistence, Same-hour-yesterday, a Static model
            (never updated) and the Online model. Use this for the results in your report.
"""
import argparse
import calendar
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)

H = 24                      # forecast horizons: 1..24 hours ahead
L_SHORT = 24                # short-term view length (hours)
N_DAYS = 7                  # daily view: 7 days x 24 hours
L_LONG = N_DAYS * 24        # 168 hours of context needed
MIN_ANCHOR = L_LONG - 1     # first usable window end index
REPORT_H = (1, 6, 12, 24)   # horizons shown in the reports
MONTH_NUM = {m: i for i, m in enumerate(calendar.month_name) if m}


# ------------------------------------------------------------------ data loading
def parse_month(path):
    """(year, month) from a file name such as ..._January_2026_Hourly.xlsx"""
    m = re.search(r"(" + "|".join(MONTH_NUM) + r")[_\- ]*(\d{4})", os.path.basename(path), re.I)
    if not m:
        raise ValueError(f"Cannot find 'MonthName_Year' in file name: {os.path.basename(path)}")
    return int(m.group(2)), MONTH_NUM[m.group(1).capitalize()]


def read_month_file(path):
    """One monthly wide file -> hourly pandas Series indexed by timestamp."""
    year, month = parse_month(path)
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    day_col = df.columns[0]
    hour_cols = [c for c in df.columns[1:] if re.match(r"^\s*\d{1,2}:\d{2}", str(c))]
    if len(hour_cols) != 24:
        raise ValueError(f"{os.path.basename(path)}: expected 24 hour columns, found {len(hour_cols)}")
    days_in_month = calendar.monthrange(year, month)[1]
    rows = []
    for _, r in df.iterrows():
        raw = r[day_col]
        d = raw.day if isinstance(raw, (pd.Timestamp, __import__("datetime").datetime)) \
            else pd.to_numeric(raw, errors="coerce")
        if pd.isna(d) or not (1 <= int(d) <= days_in_month):
            continue
        for c in hour_cols:
            hh = int(str(c).strip().split(":")[0])
            v = pd.to_numeric(r[c], errors="coerce")
            rows.append((pd.Timestamp(year, month, int(d), hh), v))
    s = pd.Series(dict(rows)).sort_index().astype(float)
    full = pd.date_range(pd.Timestamp(year, month, 1), periods=days_in_month * 24, freq="h")
    return s.reindex(full)


def list_files(data_dir, pattern):
    files = sorted(glob.glob(os.path.join(data_dir, pattern)), key=parse_month)
    if not files:
        sys.exit(f"No files matching {pattern!r} in {data_dir!r}")
    return files


def fill_gaps(s):
    """Interpolate short gaps; warn about long ones."""
    n_missing = int(s.isna().sum())
    if n_missing:
        print(f"  [data] {n_missing} missing hourly values -> interpolated")
    return s.interpolate(limit=12, limit_direction="both").ffill().bfill()


# ------------------------------------------------------------------ windows
def make_batch(y, hours, anchors):
    """Build model inputs for the given anchor indices (index of the 'current' hour)."""
    a = np.asarray(anchors)
    cur = y[a]
    short = (y[a[:, None] + np.arange(-L_SHORT + 1, 1)] - cur[:, None]) / 100.0
    long_ = (y[a[:, None] + np.arange(-L_LONG + 1, 1)] - cur[:, None]) / 100.0
    long_ = long_.reshape(len(a), 1, N_DAYS, 24)
    ang = 2 * np.pi * hours[a] / 24.0
    extra = np.stack([(cur - 250.0) / 150.0, np.sin(ang), np.cos(ang)], axis=1)
    out = dict(short=torch.tensor(short[:, None, :], dtype=torch.float32),
               daily=torch.tensor(long_, dtype=torch.float32),
               extra=torch.tensor(extra, dtype=torch.float32))
    return out, cur


def targets(y, anchors):
    a = np.asarray(anchors)
    return y[a[:, None] + np.arange(1, H + 1)]


def valid_anchors(n):
    """Anchors that have a full 168 h of context and a full 24 h of known future."""
    return np.arange(MIN_ANCHOR, n - H)


# ------------------------------------------------------------------ model
class ShortBranch(nn.Module):
    """1D CNN over the last 24 hours (keeps time order by flattening, no global pooling)."""
    def __init__(self, out=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * L_SHORT, out), nn.GELU())

    def forward(self, x):
        return self.fc(self.conv(x))


class DailyBranch(nn.Module):
    """2D CNN over the 7-day x 24-hour grid (compares the same hour across days)."""
    def __init__(self, out=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 8)))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 3 * 8, out), nn.GELU())

    def forward(self, x):
        return self.fc(self.conv(x))


class DualViewCNN(nn.Module):
    def __init__(self, variant="dual", dim=64):
        super().__init__()
        self.variant = variant
        self.short = ShortBranch(dim) if variant in ("dual", "short") else None
        self.daily = DailyBranch(dim) if variant in ("dual", "daily") else None
        n_branches = int(self.short is not None) + int(self.daily is not None)
        self.head = nn.Sequential(nn.Linear(n_branches * dim + 3, 128), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(128, H))

    def forward(self, b):
        parts = []
        if self.short is not None:
            parts.append(self.short(b["short"]))
        if self.daily is not None:
            parts.append(self.daily(b["daily"]))
        parts.append(b["extra"])
        return self.head(torch.cat(parts, dim=1))        # predicted change / 100, per horizon


def predict(model, y, hours, anchors, bs=512):
    """Absolute AQI forecasts, shape (n_anchors, 24)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(anchors), bs):
            a = anchors[i:i + bs]
            b, cur = make_batch(y, hours, a)
            d = model(b).numpy() * 100.0
            outs.append(np.clip(cur[:, None] + d, 0, 500))
    return np.concatenate(outs) if outs else np.zeros((0, H))


def fit(model, y, hours, train_anchors, epochs, lr, bs, val_anchors=None, patience=8,
        extra_anchors=None, seed=0, verbose=True):
    """Train on train_anchors (+ optional extra replay anchors). Early-stops on val if given."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    pool = np.concatenate([train_anchors, extra_anchors]) if extra_anchors is not None and len(extra_anchors) else train_anchors
    best, best_state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(pool)
        for i in range(0, len(order), bs):
            a = order[i:i + bs]
            if len(a) < 2:
                continue
            b, cur = make_batch(y, hours, a)
            tgt = torch.tensor((targets(y, a) - cur[:, None]) / 100.0, dtype=torch.float32)
            opt.zero_grad()
            loss = loss_fn(model(b), tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if val_anchors is not None and len(val_anchors):
            p = predict(model, y, hours, val_anchors)
            v = float(np.mean((p - targets(y, val_anchors)) ** 2))
            if verbose:
                print(f"    epoch {ep:02d}  val MSE = {v:8.2f}")
            if v < best - 1e-3:
                best, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ------------------------------------------------------------------ evaluation
def baselines(y, anchors):
    a = np.asarray(anchors)
    cur = y[a]
    persist = np.repeat(cur[:, None], H, axis=1)
    yesterday = y[a[:, None] + np.arange(1, H + 1) - 24]        # same hour one day earlier
    return {"Persistence": persist, "Same hour yesterday": yesterday}


def metrics(actual, pred):
    out = {}
    for h in REPORT_H:
        e = pred[:, h - 1] - actual[:, h - 1]
        out[str(h)] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}
    return out


def evaluate(y, hours, anchors, models):
    act = targets(y, anchors)
    preds = baselines(y, anchors)
    for name, m in models.items():
        preds[name] = predict(m, y, hours, anchors)
    return {k: metrics(act, v) for k, v in preds.items()}, preds, act


# ------------------------------------------------------------------ state
def paths(state_dir):
    os.makedirs(state_dir, exist_ok=True)
    return {k: os.path.join(state_dir, f) for k, f in
            dict(history="history.csv", online="online.pt", static="static.pt",
                 log="log.json", forecast="forecast.csv").items()}


def load_history(p):
    df = pd.read_csv(p["history"], parse_dates=["datetime"])
    return df["datetime"].values, df["aqi"].values.astype(np.float64)


def save_history(p, idx, y):
    pd.DataFrame({"datetime": pd.to_datetime(idx), "aqi": y}).to_csv(p["history"], index=False)


def hours_of(idx):
    return pd.DatetimeIndex(idx).hour.values.astype(np.float64)


def save_model(path, model, variant, trained_upto):
    torch.save({"state": model.state_dict(), "variant": variant, "trained_upto": int(trained_upto)}, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = DualViewCNN(ck["variant"])
    m.load_state_dict(ck["state"])
    return m, ck["variant"], ck["trained_upto"]


def write_forecast(p, model, idx, y):
    a = np.array([len(y) - 1])
    pred = predict(model, y, hours_of(idx), a)[0]
    t0 = pd.Timestamp(idx[-1])
    df = pd.DataFrame({"datetime": [t0 + pd.Timedelta(hours=h) for h in range(1, H + 1)],
                       "horizon_h": range(1, H + 1), "forecast_aqi": np.round(pred, 1)})
    df.to_csv(p["forecast"], index=False)
    return df


# ------------------------------------------------------------------ core steps
def build_series(files):
    parts = [read_month_file(f) for f in files]
    for f, s in zip(files, parts):
        print(f"  {os.path.basename(f)}: {len(s)} hours")
    s = pd.concat(parts)
    full = pd.date_range(s.index[0], s.index[-1], freq="h")
    s = s.reindex(full)
    return fill_gaps(s)


def train_base(y, hours, variant, epochs, seed=0, verbose=True):
    torch.manual_seed(seed)
    anchors = valid_anchors(len(y))
    n_val = max(int(0.15 * len(anchors)), 24)
    tr = anchors[:len(anchors) - n_val - H]           # gap so train targets never reach validation
    va = anchors[len(anchors) - n_val:]
    if len(tr) < 100:
        sys.exit("Not enough data to train. Give at least ~1.5 months for the first training.")
    model = DualViewCNN(variant)
    fit(model, y, hours, tr, epochs=epochs, lr=2e-3, bs=128, val_anchors=va, seed=seed, verbose=verbose)
    return model, anchors[-1]


def online_update(model, y, hours, trained_upto, epochs, replay_ratio, seed=0):
    """Fine-tune on anchors not yet trained on, mixed with a replay sample of older anchors."""
    new = np.arange(trained_upto + 1, len(y) - H)
    if len(new) == 0:
        return model, trained_upto
    old = np.arange(MIN_ANCHOR, trained_upto + 1)
    rng = np.random.default_rng(seed)
    k = min(len(old), int(len(new) * replay_ratio))
    replay = rng.choice(old, size=k, replace=False) if k else np.array([], dtype=int)
    torch.manual_seed(seed)
    fit(model, y, hours, new, epochs=epochs, lr=5e-4, bs=64, extra_anchors=replay, seed=seed, verbose=False)
    return model, new[-1]


def cmd_init(a):
    p = paths(a.state_dir)
    files = list_files(a.data_dir, a.pattern)
    print(f"Reading {len(files)} month file(s):")
    s = build_series(files)
    y, hours = s.values.astype(np.float64), hours_of(s.index)
    print(f"History: {len(y)} hours ({s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d})")
    print(f"Training base model ({a.variant}) ...")
    model, upto = train_base(y, hours, a.variant, a.epochs, a.seed)
    save_history(p, s.index.values, y)
    save_model(p["online"], model, a.variant, upto)
    save_model(p["static"], model, a.variant, upto)
    json.dump([], open(p["log"], "w"))
    df = write_forecast(p, model, s.index.values, y)
    print("\nNext 24 hours (from the latest hour in your data):")
    print(df.to_string(index=False))
    print(f"\nSaved state in {a.state_dir}/")


def cmd_update(a):
    p = paths(a.state_dir)
    idx, y = load_history(p)
    online, variant, upto = load_model(p["online"])
    static, _, _ = load_model(p["static"])
    year, month = parse_month(a.file)
    new_start = pd.Timestamp(year, month, 1)
    if new_start <= pd.Timestamp(idx[-1]):
        sys.exit(f"{os.path.basename(a.file)} ({year}-{month:02d}) is already in the history. Nothing to do.")
    new = read_month_file(a.file)
    # join to existing history, filling any gap between the months
    s_old = pd.Series(y, index=pd.DatetimeIndex(idx))
    s = fill_gaps(pd.concat([s_old, new]).reindex(pd.date_range(s_old.index[0], new.index[-1], freq="h")))
    y_all, hours_all = s.values.astype(np.float64), hours_of(s.index)
    n_prev = len(y)

    # 1) honest forecast of the new month with the models as they are NOW
    anchors = np.arange(max(MIN_ANCHOR, n_prev), len(y_all) - H)
    res, _, _ = evaluate(y_all, hours_all, anchors, {"Static model": static, "Online model": online})
    print(f"\nForecast accuracy on {year}-{month:02d} BEFORE learning it ({len(anchors)} test hours):")
    print_metrics({"Persistence": res["Persistence"], "Same hour yesterday": res["Same hour yesterday"],
                   "Static model": res["Static model"], "Online model": res["Online model"]})

    # 2) learn from the new month
    online, upto = online_update(online, y_all, hours_all, upto, a.epochs, a.replay_ratio, a.seed)
    save_history(p, s.index.values, y_all)
    save_model(p["online"], online, variant, upto)
    log = json.load(open(p["log"]))
    log.append({"month": f"{year}-{month:02d}", "n_hours": len(anchors), "metrics": res})
    json.dump(log, open(p["log"], "w"), indent=1)

    df = write_forecast(p, online, s.index.values, y_all)
    print("\nModel updated. Next 24 hours (from the latest hour in your data):")
    print(df.to_string(index=False))


def cmd_forecast(a):
    p = paths(a.state_dir)
    idx, y = load_history(p)
    model, _, _ = load_model(p["online"])
    print(write_forecast(p, model, idx, y).to_string(index=False))


def print_metrics(res):
    rows = []
    for name, m in res.items():
        row = {"Model": name}
        for h in REPORT_H:
            row[f"RMSE {h}h"] = round(m[str(h)]["rmse"], 1)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_simulate(a):
    files = list_files(a.data_dir, a.pattern)
    if len(files) <= a.init_months:
        sys.exit(f"Need more than {a.init_months} month files to simulate (found {len(files)}).")
    print(f"Reading {len(files)} month files ...")
    full = build_series(files)
    month_of = full.index.to_period("M")
    months = list(dict.fromkeys(month_of))
    init_end = full.index[month_of == months[a.init_months - 1]][-1]
    print(f"Initial training: {months[0]} .. {months[a.init_months - 1]}  |  "
          f"simulated stream: {months[a.init_months]} .. {months[-1]}")
    variants = [v.strip() for v in a.variants.split(",")]
    summary = {}
    base_res = {}
    for variant in variants:
        print(f"\n===== variant: {variant} =====")
        init_n = int((full.index <= init_end).sum())
        y0, h0 = full.values[:init_n].astype(np.float64), hours_of(full.index[:init_n])
        static, upto = train_base(y0, h0, variant, a.epochs, a.seed, verbose=False)
        online = DualViewCNN(variant)
        online.load_state_dict(static.state_dict())
        log = []
        for m in months[a.init_months:]:
            end_n = int((full.index <= full.index[month_of == m][-1]).sum())
            y, hrs = full.values[:end_n].astype(np.float64), hours_of(full.index[:end_n])
            n_prev = int((full.index < full.index[month_of == m][0]).sum())
            anchors = np.arange(max(MIN_ANCHOR, n_prev), len(y) - H)
            res, _, _ = evaluate(y, hrs, anchors, {"Static": static, "Online": online})
            log.append({"month": str(m), "n_hours": int(len(anchors)), "metrics": res})
            online, upto = online_update(online, y, hrs, upto, a.update_epochs, a.replay_ratio, a.seed)
            print(f"  {m}: Persistence RMSE@6h={res['Persistence']['6']['rmse']:.1f} | "
                  f"Static={res['Static']['6']['rmse']:.1f} | Online={res['Online']['6']['rmse']:.1f}")
        # weighted average over simulated months
        w = np.array([r["n_hours"] for r in log], dtype=float)
        for name in log[0]["metrics"]:
            avg = {str(h): {"rmse": float(np.sqrt(np.average([r["metrics"][name][str(h)]["rmse"] ** 2 for r in log], weights=w))),
                            "mae": float(np.average([r["metrics"][name][str(h)]["mae"] for r in log], weights=w))}
                   for h in REPORT_H}
            if name in ("Persistence", "Same hour yesterday"):
                base_res[name] = avg
            else:
                summary[f"{name} ({variant})"] = avg
        out_dir = os.path.join(a.state_dir, f"sim_{variant}")
        os.makedirs(out_dir, exist_ok=True)
        json.dump(log, open(os.path.join(out_dir, "log.json"), "w"), indent=1)
    final = {**base_res, **summary}
    print("\n=== Average RMSE over the simulated months (lower is better) ===")
    print_metrics(final)
    json.dump({"summary": final}, open(os.path.join(a.state_dir, "simulation_summary.json"), "w"), indent=1)
    print(f"\nSaved logs in {a.state_dir}/")


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--state_dir", default="aqi_state")
        p.add_argument("--seed", type=int, default=42)

    def data_args(p):
        p.add_argument("--data_dir", default=".")
        p.add_argument("--pattern", default="*City_Level*.xlsx")

    p = sub.add_parser("init"); common(p); data_args(p)
    p.add_argument("--variant", choices=["dual", "short", "daily"], default="dual")
    p.add_argument("--epochs", type=int, default=60)
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("update"); common(p)
    p.add_argument("--file", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--replay_ratio", type=float, default=1.0)
    p.set_defaults(fn=cmd_update)

    p = sub.add_parser("forecast"); common(p)
    p.set_defaults(fn=cmd_forecast)

    p = sub.add_parser("simulate"); common(p); data_args(p)
    p.add_argument("--init_months", type=int, default=4)
    p.add_argument("--variants", default="dual,short,daily")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--update_epochs", type=int, default=8)
    p.add_argument("--replay_ratio", type=float, default=1.0)
    p.set_defaults(fn=cmd_simulate)

    args = ap.parse_args()
    args.fn(args)
