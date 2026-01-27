# stepF_sign_event_linear.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

try:
    from ._path_setup import SRC  # noqa: F401
except ImportError:
    from _path_setup import SRC  # noqa: F401
from run_utils import resolve_run_dir, build_run_paths, repo_root


def pivot_wide(df: pd.DataFrame, value_cols, prefix=""):
    wide = df.pivot_table(index="epoch", columns="island", values=value_cols, aggfunc="mean")
    wide.columns = [f"{prefix}{col}_island{isl}" for col, isl in wide.columns]
    wide = wide.reset_index()
    for col in value_cols:
        c0 = f"{prefix}{col}_island0"
        c1 = f"{prefix}{col}_island1"
        if c0 in wide.columns and c1 in wide.columns:
            wide[f"{prefix}{col}_diff_0_minus_1"] = wide[c0] - wide[c1]
    return wide


def safe_numeric_cols(df: pd.DataFrame, exclude=("epoch",)):
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def loo_prob_predictions(model, X, y, pos_label=1):
    n = len(y)
    prob = np.zeros(n, dtype=float)
    pred = np.zeros(n, dtype=int)

    for i in range(n):
        idx_tr = np.array([j for j in range(n) if j != i], dtype=int)
        y_tr = y[idx_tr]
        uniq = np.unique(y_tr)

        if len(uniq) < 2:
            c = int(uniq[0])
            pred[i] = c
            prob[i] = 1.0 if c == pos_label else 0.0
            continue

        model.fit(X[idx_tr], y_tr)
        p = model.predict_proba(X[i:i+1])[0]
        classes = model.classes_
        pos_idx = int(np.where(classes == pos_label)[0][0])
        prob[i] = p[pos_idx]
        pred[i] = int(prob[i] >= 0.5)

    return prob, pred


def plot_top_coeffs(coef_series: pd.Series, title: str, outpath: Path, topk=20):
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--env", type=str, default=None)
    ap.add_argument("--profile", type=str, default=None)
    ap.add_argument("--thr-ts", type=str, default=None)
    ap.add_argument("--events", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args(argv)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)

    env_path = Path(args.env) if args.env else (paths.derived_dir / "island_env_error.csv")
    prof_path = Path(args.profile) if args.profile else (paths.derived_dir / "island_profile.csv")
    thr_ts_path = Path(args.thr_ts) if args.thr_ts else (paths.derived_dir / "threshold_timeseries_with_events.csv")
    events_path = Path(args.events) if args.events else (paths.derived_dir / "threshold_update_events.csv")

    if not env_path.is_absolute():
        env_path = repo_root() / env_path
    if not prof_path.is_absolute():
        prof_path = repo_root() / prof_path
    if not thr_ts_path.is_absolute():
        thr_ts_path = repo_root() / thr_ts_path
    if not events_path.is_absolute():
        events_path = repo_root() / events_path

    out_dir = Path(args.out_dir) if args.out_dir else (paths.derived_dir / "stepF_sign_event_linear")
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = pd.read_csv(env_path)
    prof = pd.read_csv(prof_path)
    thr_ts = pd.read_csv(thr_ts_path)
    events = pd.read_csv(events_path)

    event_sign_map = dict(zip(events["epoch"].astype(int), events.get("event_sign", pd.Series(dtype=int)).astype(int)))

    env_cols = [c for c in env.columns if c not in ("epoch", "island", "n")]
    prof_cols = [c for c in prof.columns if c not in ("epoch", "island", "n")]

    env_w = pivot_wide(env, env_cols, prefix="env__")
    prof_w = pivot_wide(prof, prof_cols, prefix="prof__")

    feat = env_w.merge(prof_w, on="epoch", how="inner")
    feat = feat.merge(thr_ts[["epoch", "threshold", "is_event"]], on="epoch", how="left")
    feat = feat.sort_values("epoch").reset_index(drop=True)

    epochs = feat["epoch"].astype(int).tolist()
    rows = []
    for i in range(1, len(epochs)):
        e_prev = epochs[i - 1]
        e = epochs[i]
        prev_row = feat.loc[feat["epoch"] == e_prev].iloc[0]
        cur_row = feat.loc[feat["epoch"] == e].iloc[0]

        y_event = int(bool(cur_row["is_event"])) if not pd.isna(cur_row["is_event"]) else 0
        y_sign = event_sign_map.get(e, np.nan)

        r = {"epoch_prev": e_prev, "epoch": e, "y_event": y_event, "y_sign": y_sign}
        for c in safe_numeric_cols(feat, exclude=("epoch",)):
            r[f"Xprev__{c}"] = float(prev_row[c]) if pd.notna(prev_row[c]) else np.nan
        rows.append(r)

    ds = pd.DataFrame(rows)
    x_cols = [c for c in ds.columns if c.startswith("Xprev__")]
    ds_clean = ds.dropna(subset=x_cols).reset_index(drop=True)

    X = ds_clean[x_cols].values
    y_event = ds_clean["y_event"].astype(int).values

    event_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=10000,
        )),
    ])

    prob_event, pred_event = loo_prob_predictions(event_model, X, y_event)
    acc_event = accuracy_score(y_event, pred_event)
    try:
        auc_event = roc_auc_score(y_event, prob_event)
    except Exception:
        auc_event = float("nan")
    cm_event = confusion_matrix(y_event, pred_event)

    event_model.fit(X, y_event)
    coef_event = event_model.named_steps["clf"].coef_.ravel()
    coef_event_s = pd.Series(coef_event, index=x_cols).sort_values(key=lambda s: s.abs(), ascending=False)

    pd.DataFrame({
        "feature": coef_event_s.index,
        "coef": coef_event_s.values,
        "abs_coef": np.abs(coef_event_s.values),
    }).to_csv(out_dir / "stepF_event_binary_coeffs.csv", index=False)

    with open(out_dir / "stepF_event_binary_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"LOO accuracy: {acc_event:.6f}\n")
        f.write(f"LOO ROC-AUC : {auc_event:.6f}\n")
        f.write("Confusion matrix [[TN, FP],[FN, TP]]:\n")
        f.write(str(cm_event) + "\n")

    plot_top_coeffs(
        coef_event_s,
        title="Event (0/1) logistic: top coefficients (Xprev)",
        outpath=out_dir / "stepF_event_binary_topcoef.png",
        topk=args.topk,
    )

    ds_evt = ds_clean[ds_clean["y_event"] == 1].copy()
    ds_evt = ds_evt.dropna(subset=["y_sign"]).reset_index(drop=True)
    if len(ds_evt) >= 4 and ds_evt["y_sign"].nunique() >= 2:
        Xs = ds_evt[x_cols].values
        y_sign_bin = (ds_evt["y_sign"].astype(int).values == 1).astype(int)

        sign_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                solver="liblinear",
                class_weight="balanced",
                max_iter=10000,
            )),
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
            "abs_coef": np.abs(coef_sign_s.values),
        }).to_csv(out_dir / "stepF_event_sign_coeffs.csv", index=False)

        with open(out_dir / "stepF_event_sign_metrics.txt", "w", encoding="utf-8") as f:
            f.write("Sign model is trained on EVENT rows only.\n")
            f.write("Target: +1=1, -1=0\n")
            f.write(f"LOO accuracy: {acc_sign:.6f}\n")
            f.write(f"LOO ROC-AUC : {auc_sign:.6f}\n")
            f.write("Confusion matrix [[TN, FP],[FN, TP]]:\n")
            f.write(str(cm_sign) + "\n")

        plot_top_coeffs(
            coef_sign_s,
            title="Event SIGN (+1 vs -1) logistic: top coefficients (Xprev, among events)",
            outpath=out_dir / "stepF_event_sign_topcoef.png",
            topk=args.topk,
        )
    else:
        with open(out_dir / "stepF_event_sign_metrics.txt", "w", encoding="utf-8") as f:
            f.write("Not enough event rows or sign variety to train sign model.\n")
            f.write(
                f"n_event_rows={len(ds_evt)}, "
                f"unique_signs={ds_evt['y_sign'].nunique() if 'y_sign' in ds_evt else 'NA'}\n"
            )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
