# stepF_sign_event_linear.py
# - 直前状態 Xprev だけで
#   (A) 次で event が起きるか (0/1)
#   (B) event が起きたら sign が +1(↑) / -1(↓) か
# を線形（Logistic回帰）で説明する。
#
# 入力:
#   island_env_error.csv
#   island_profile.csv
#   threshold_timeseries_with_events.csv  (epoch, threshold, is_event が入っている想定)
#   threshold_update_events.csv           (event_sign が入っている想定)
#
# 出力: stepF_sign_event_linear/*
#   - stepF_event_binary_coeffs.csv
#   - stepF_event_sign_coeffs.csv
#   - stepF_event_binary_metrics.txt
#   - stepF_event_sign_metrics.txt
#   - stepF_event_binary_topcoef.png
#   - stepF_event_sign_topcoef.png

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt


# ========= I/O =========
ENV_PATH  = "/content/island_env_error.csv"
PROF_PATH = "/content/island_profile.csv"
THR_TS_PATH = "/content/threshold_timeseries_with_events.csv"
EVENTS_PATH = "/content/threshold_update_events.csv"

OUTDIR = "stepF_sign_event_linear"
os.makedirs(OUTDIR, exist_ok=True)


# ========= helpers =========
def pivot_wide(df: pd.DataFrame, value_cols, prefix=""):
    """epoch×island を wide にして、island0/1列 + diff(0-1) を作る"""
    wide = df.pivot_table(index="epoch", columns="island", values=value_cols, aggfunc="mean")
    wide.columns = [f"{prefix}{col}_island{isl}" for col, isl in wide.columns]
    wide = wide.reset_index()
    for col in value_cols:
        c0=f"{prefix}{col}_island0"
        c1=f"{prefix}{col}_island1"
        if c0 in wide.columns and c1 in wide.columns:
            wide[f"{prefix}{col}_diff_0_minus_1"] = wide[c0] - wide[c1]
    return wide

def safe_numeric_cols(df: pd.DataFrame, exclude=("epoch",)):
    cols=[]
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def loo_prob_predictions(model, X, y, pos_label=1):
    """
    LOOで確率予測を返す。train側が単一クラスになるfoldは、
    そのクラスで確定予測にフォールバックする。
    """
    n = len(y)
    prob = np.zeros(n, dtype=float)
    pred = np.zeros(n, dtype=int)

    for i in range(n):
        idx_te = i
        idx_tr = np.array([j for j in range(n) if j != i], dtype=int)

        y_tr = y[idx_tr]
        uniq = np.unique(y_tr)

        if len(uniq) < 2:
            # trainが単一クラス → そのクラスで確定
            c = int(uniq[0])
            pred[idx_te] = c
            prob[idx_te] = 1.0 if c == pos_label else 0.0
            continue

        model.fit(X[idx_tr], y_tr)
        p = model.predict_proba(X[idx_te:idx_te+1])[0]
        # pos_label の列を取る（pos_labelが1でない場合も安全に）
        classes = model.classes_
        pos_idx = int(np.where(classes == pos_label)[0][0])
        prob[idx_te] = p[pos_idx]
        pred[idx_te] = int(prob[idx_te] >= 0.5)

    return prob, pred

def loo_pred_multiclass(model, X, y):
    """LOO でクラス予測（multiclass用）"""
    n = X.shape[0]
    pred = np.full(n, -999, dtype=int)
    for i in range(n):
        idx_tr = np.array([j for j in range(n) if j != i])
        idx_te = np.array([i])
        model.fit(X[idx_tr], y[idx_tr])
        pred[i] = int(model.predict(X[idx_te])[0])
    return pred

def plot_top_coeffs(coef_series: pd.Series, title: str, outpath: str, topk=20):
    s = coef_series.copy()
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    s = s.head(topk)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(s)), s.values)
    plt.yticks(range(len(s)), s.index, fontsize=8)
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ========= load =========
env  = pd.read_csv(ENV_PATH)
prof = pd.read_csv(PROF_PATH)
thr_ts = pd.read_csv(THR_TS_PATH)
events = pd.read_csv(EVENTS_PATH)

# event_sign を epoch に紐づけ（event の時だけ存在）
event_sign_map = dict(zip(events["epoch"].astype(int), events["event_sign"].astype(int)))

# ========= build epoch-level feature table =========
env_cols  = [c for c in env.columns  if c not in ("epoch","island","n")]
prof_cols = [c for c in prof.columns if c not in ("epoch","island","n")]

env_w  = pivot_wide(env,  env_cols,  prefix="env__")
prof_w = pivot_wide(prof, prof_cols, prefix="prof__")

feat = env_w.merge(prof_w, on="epoch", how="inner")
feat = feat.merge(thr_ts[["epoch","threshold","is_event"]], on="epoch", how="left")

feat = feat.sort_values("epoch").reset_index(drop=True)

# ========= make (Xprev -> y_event, y_sign) dataset =========
epochs = feat["epoch"].astype(int).tolist()

rows=[]
for i in range(1, len(epochs)):
    e_prev = epochs[i-1]
    e      = epochs[i]
    prev_row = feat.loc[feat["epoch"]==e_prev].iloc[0]
    cur_row  = feat.loc[feat["epoch"]==e].iloc[0]

    # y_event: 現epochで event?
    y_event = int(bool(cur_row["is_event"])) if not pd.isna(cur_row["is_event"]) else 0

    # y_sign: event のときだけ +1/-1 を付与（なければ NaN）
    y_sign = event_sign_map.get(e, np.nan)

    r = {"epoch_prev": e_prev, "epoch": e, "y_event": y_event, "y_sign": y_sign}

    # Xprev: 直前状態特徴量（epoch以外の数値全部）
    for c in safe_numeric_cols(feat, exclude=("epoch",)):
        r[f"Xprev__{c}"] = float(prev_row[c]) if pd.notna(prev_row[c]) else np.nan

    rows.append(r)

ds = pd.DataFrame(rows)

# 欠損を落とす（線形最速のため。必要なら後で補完に変更可）
x_cols = [c for c in ds.columns if c.startswith("Xprev__")]
ds_clean = ds.dropna(subset=x_cols).reset_index(drop=True)

X = ds_clean[x_cols].values
y_event = ds_clean["y_event"].astype(int).values

# ========= Model (A): event 0/1 =========
# 小標本なので強い正則化（C小さめ）+ class_weight
event_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=10000
    ))
])

prob_event, pred_event = loo_prob_predictions(event_model, X, y_event)
acc_event = accuracy_score(y_event, pred_event)

# ROC-AUC は両クラス存在しないと計算できない
try:
    auc_event = roc_auc_score(y_event, prob_event)
except Exception:
    auc_event = float("nan")

cm_event = confusion_matrix(y_event, pred_event)

# fit full to export coeffs
event_model.fit(X, y_event)
coef_event = event_model.named_steps["clf"].coef_.ravel()
coef_event_s = pd.Series(coef_event, index=x_cols).sort_values(key=lambda s: s.abs(), ascending=False)

pd.DataFrame({
    "feature": coef_event_s.index,
    "coef": coef_event_s.values,
    "abs_coef": np.abs(coef_event_s.values)
}).to_csv(f"{OUTDIR}/stepF_event_binary_coeffs.csv", index=False)

with open(f"{OUTDIR}/stepF_event_binary_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"LOO accuracy: {acc_event:.6f}\n")
    f.write(f"LOO ROC-AUC : {auc_event:.6f}\n")
    f.write("Confusion matrix [[TN, FP],[FN, TP]]:\n")
    f.write(str(cm_event) + "\n")

plot_top_coeffs(
    coef_event_s,
    title="Event (0/1) logistic: top coefficients (Xprev)",
    outpath=f"{OUTDIR}/stepF_event_binary_topcoef.png",
    topk=25
)

# ========= Model (B): sign (+1/-1) among events =========
ds_evt = ds_clean[ds_clean["y_event"]==1].copy()
ds_evt = ds_evt.dropna(subset=["y_sign"]).reset_index(drop=True)
if len(ds_evt) >= 4 and ds_evt["y_sign"].nunique() >= 2:
    Xs = ds_evt[x_cols].values
    # map: -1 -> 0, +1 -> 1
    y_sign_bin = (ds_evt["y_sign"].astype(int).values == 1).astype(int)

    sign_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=10000
        ))
    ])

    prob_sign, pred_sign = loo_prob_predictions(sign_model, Xs, y_sign_bin)
    acc_sign = accuracy_score(y_sign_bin, pred_sign)
    try:
        auc_sign = roc_auc_score(y_sign_bin, prob_sign)
    except Exception:
        auc_sign = float("nan")
    cm_sign = confusion_matrix(y_sign_bin, pred_sign)

    sign_model.fit(Xs, y_sign_bin)
    coef_sign = sign_model.named_steps["clf"].coef_.ravel()
    coef_sign_s = pd.Series(coef_sign, index=x_cols).sort_values(key=lambda s: s.abs(), ascending=False)

    pd.DataFrame({
        "feature": coef_sign_s.index,
        "coef": coef_sign_s.values,
        "abs_coef": np.abs(coef_sign_s.values)
    }).to_csv(f"{OUTDIR}/stepF_event_sign_coeffs.csv", index=False)

    with open(f"{OUTDIR}/stepF_event_sign_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Sign model is trained on EVENT rows only.\n")
        f.write("Target: +1(↑)=1, -1(↓)=0\n")
        f.write(f"LOO accuracy: {acc_sign:.6f}\n")
        f.write(f"LOO ROC-AUC : {auc_sign:.6f}\n")
        f.write("Confusion matrix [[TN, FP],[FN, TP]] (TN means predicted ↓ when true ↓):\n")
        f.write(str(cm_sign) + "\n")

    plot_top_coeffs(
        coef_sign_s,
        title="Event SIGN (↑ vs ↓) logistic: top coefficients (Xprev, among events)",
        outpath=f"{OUTDIR}/stepF_event_sign_topcoef.png",
        topk=25
    )
else:
    # 保存だけして終わる
    with open(f"{OUTDIR}/stepF_event_sign_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Not enough event rows or not enough sign variety to train sign model.\n")
        f.write(f"n_event_rows={len(ds_evt)}, unique_signs={ds_evt['y_sign'].nunique() if 'y_sign' in ds_evt else 'NA'}\n")

print(f"Saved outputs to: {OUTDIR}")
print("Binary event model: see stepF_event_binary_metrics.txt / coeffs.csv / png")
print("Sign model (events only): see stepF_event_sign_metrics.txt / coeffs.csv / png (if trained)")
