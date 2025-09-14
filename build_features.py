#!/usr/bin/env python3
"""
Build 196-dim feature datasets from official UNSW-NB15 49-column CSVs, split into 5 client sets,
and save a consistent MinMaxScaler and feature spec for live detection.

Inputs (with headers):
  Training and Testing Sets/UNSW_NB15_training-set.csv
  Training and Testing Sets/UNSW_NB15_testing-set.csv

Outputs:
  data/UNSW_NB15_Train201.csv ... data/UNSW_NB15_Train205.csv  (each row: 196 features + label)
  data/UNSW_NB15_TestBin.csv                                   (196 features + label)
  CentralServer/scaler.pkl                                      (MinMaxScaler fitted on training features)
  CentralServer/feature_spec.json                               (frozen feature order and encoding)

NOTE: These features must also be computed in the live detector to ensure alignment.
"""

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold


# --------- Config ---------
TRAIN_CSV = os.path.join("Training and Testing Sets", "UNSW_NB15_training-set.csv")
TEST_CSV = os.path.join("Training and Testing Sets", "UNSW_NB15_testing-set.csv")

OUT_DIR = os.path.join("data", "NewData")
CENTRAL_DIR = "CentralServer"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CENTRAL_DIR, exist_ok=True)

# One-hot vocab sizes (cap to keep total dims <= 196)
MAX_SERVICE = 50  # top-k services by frequency, others -> "other"


def load_unsw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    return train, test


def build_vocab(train: pd.DataFrame) -> Dict:
    """Build categorical vocabularies from training only (frozen for test/live)."""
    vocab: Dict[str, List[str]] = {}

    # proto: include all observed
    vocab_proto = sorted(train["proto"].astype(str).fillna("unknown").unique().tolist())
    vocab["proto"] = vocab_proto

    # state: include all observed
    vocab_state = sorted(train["state"].astype(str).fillna("unknown").unique().tolist())
    vocab["state"] = vocab_state

    # service: take top-K by frequency
    svc_counts = (
        train["service"].astype(str).fillna("unknown").value_counts(dropna=False)
    )
    top_services = svc_counts.head(MAX_SERVICE).index.astype(str).tolist()
    if "other" not in top_services:
        top_services.append("other")
    vocab["service_top"] = top_services

    return vocab


def _one_hot(value: str, vocab: List[str]) -> List[int]:
    vec = [0] * len(vocab)
    try:
        idx = vocab.index(value)
    except ValueError:
        idx = -1
    if idx >= 0:
        vec[idx] = 1
    return vec


def row_to_features(row: pd.Series, vocab: Dict, feature_names: List[str]) -> np.ndarray:
    """Construct a 196-dim numeric vector from a single row.

    This mapping is intentionally deterministic and reproducible from the 49 UNSW columns.
    It combines numeric fields + one-hots + simple engineered ratios until 196 dims are reached.
    The exact names/order are stored in feature_spec.json for use in live detector.
    """
    feats: List[float] = []
    names: List[str] = []

    # Numeric base (exclude id, attack_cat, label, categoricals)
    numeric_cols = [
        "dur","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl","sload","dload",
        "sloss","dloss","sinpkt","dinpkt","sjit","djit","swin","stcpb","dtcpb","dwin",
        "tcprtt","synack","ackdat","smean","dmean","trans_depth","response_body_len",
        "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
        "ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_flw_http_mthd","ct_src_ltm","ct_srv_dst",
        "is_sm_ips_ports",
    ]
    for col in numeric_cols:
        val = row.get(col, 0)
        try:
            feats.append(float(val))
        except Exception:
            feats.append(0.0)
        names.append(col)

    # Categorical: proto
    proto = str(row.get("proto", "unknown"))
    proto_vec = _one_hot(proto, vocab["proto"])
    feats.extend(proto_vec)
    names.extend([f"proto=={p}" for p in vocab["proto"]])

    # Categorical: state
    state = str(row.get("state", "unknown"))
    state_vec = _one_hot(state, vocab["state"])
    feats.extend(state_vec)
    names.extend([f"state=={s}" for s in vocab["state"]])

    # Categorical: service (top-K + other)
    service_raw = str(row.get("service", "unknown"))
    service = service_raw if service_raw in vocab["service_top"] else "other"
    service_vec = _one_hot(service, vocab["service_top"])
    feats.extend(service_vec)
    names.extend([f"service=={s}" for s in vocab["service_top"]])

    # Simple engineered ratios (deterministic)
    # Guard against division by zero
    def safe_div(a, b):
        try:
            a = float(a)
            b = float(b)
            return (a / b) if b not in (0.0, 0) else 0.0
        except Exception:
            return 0.0

    sbytes = row.get("sbytes", 0)
    dbytes = row.get("dbytes", 0)
    spkts = row.get("spkts", 0)
    dpkts = row.get("dpkts", 0)
    rate = row.get("rate", 0)
    dur = row.get("dur", 0)
    smean = row.get("smean", 0)
    dmean = row.get("dmean", 0)

    engineered: List[Tuple[str, float]] = [
        ("ratio_sbytes_dbytes", safe_div(sbytes, dbytes)),
        ("ratio_dbytes_sbytes", safe_div(dbytes, sbytes)),
        ("ratio_spkts_dpkts", safe_div(spkts, dpkts)),
        ("ratio_dpkts_spkts", safe_div(dpkts, spkts)),
        ("bytes_per_dur", safe_div(float(sbytes) + float(dbytes), dur)),
        ("pkts_per_dur", safe_div(float(spkts) + float(dpkts), dur)),
        ("rate_times_dur", safe_div(rate, 1.0) * float(dur or 0.0)),
        ("ratio_smean_dmean", safe_div(smean, dmean)),
        ("ratio_dmean_smean", safe_div(dmean, smean)),
    ]

    for n, v in engineered:
        feats.append(float(v))
        names.append(n)

    # Pad with zeros if fewer than 196; trim if exceeding 196
    if len(feats) < 196:
        pad_n = 196 - len(feats)
        feats.extend([0.0] * pad_n)
        names.extend([f"pad_{i}" for i in range(pad_n)])
    elif len(feats) > 196:
        feats = feats[:196]
        names = names[:196]

    # record names encountered for the first row only
    for idx, nm in enumerate(names):
        feature_names[idx] = feature_names.get(idx, nm)

    return np.asarray(feats, dtype=np.float32)


def build_feature_matrix(df: pd.DataFrame, vocab: Dict) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    feature_names: Dict[int, str] = {}
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for _, row in df.iterrows():
        feats = row_to_features(row, vocab, feature_names)
        X_list.append(feats)
        y_list.append(int(row.get("label", 0)))

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, feature_names


def save_feature_spec(feature_names: Dict[int, str], vocab: Dict):
    spec = {
        "feature_order": [feature_names[i] for i in range(196)],
        "vocab": {
            "proto": vocab["proto"],
            "state": vocab["state"],
            "service_top": vocab["service_top"],
        },
        "tl": 4,
        "dim": 196,
    }
    with open(os.path.join(CENTRAL_DIR, "feature_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)


def write_csv(path: str, X: np.ndarray, y: np.ndarray):
    # Concatenate features and label as last column
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    np.savetxt(path, data, delimiter=",", fmt="%.6f")


def split_five_clients(X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Stratified split into 5 approximately equal folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = []
    for _, idx in skf.split(X, y):
        splits.append((X[idx], y[idx]))
    return splits


def main():
    print("📦 Loading UNSW-NB15 datasets...")
    train, test = load_unsw()

    # Build categorical vocab from training
    print("🔤 Building vocabularies from training set...")
    vocab = build_vocab(train)

    # Build features
    print("🧮 Building 196-dim features for training set...")
    X_train, y_train, feat_names = build_feature_matrix(train, vocab)
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")

    print("🧮 Building 196-dim features for test set...")
    X_test, y_test, _ = build_feature_matrix(test, vocab)
    print(f"   Test:  X={X_test.shape}, y={y_test.shape}")

    # Fit single scaler on training features only and save
    print("📏 Fitting MinMaxScaler on training features and saving scaler.pkl...")
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    import pickle
    with open(os.path.join(CENTRAL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("📝 Saving feature_spec.json...")
    save_feature_spec(feat_names, vocab)

    # For now, write UN-SCALED features to CSVs to match the current training code shape (196 + label)
    # We will update training to load and use scaler.pkl for consistency with live detector.

    print("🔀 Splitting training into 5 client datasets...")
    client_splits = split_five_clients(X_train, y_train)
    for i, (Xi, yi) in enumerate(client_splits, start=1):
        out_path = os.path.join(OUT_DIR, f"UNSW_NB15_Train20{i:01d}.csv")
        write_csv(out_path, Xi, yi)
        print(f"   Wrote {out_path} -> {Xi.shape[0]} rows")

    print("💾 Writing transformed test set to data/UNSW_NB15_TestBin.csv ...")
    write_csv(os.path.join(OUT_DIR, "UNSW_NB15_TestBin.csv"), X_test, y_test)

    print("✅ Done. Next steps:\n"
          "  1) Update training to use CentralServer/scaler.pkl (no refit).\n"
          "  2) Implement the same feature_spec.json mapping in the live detector.\n"
          )


if __name__ == "__main__":
    main()


