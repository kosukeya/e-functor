# train.py
import copy
import torch
import torch.nn.functional as F
from typing import Dict, Any

import config as C
from data import generate_world_data, make_split, pack_data
from model import MultiIWorldModel, FStatWrapper
from losses import (
    loss_real_with_attn_F2, loss_real_sem_log,
    loss_counterfactual_sun0_sem, loss_counterfactual_plant0_sem,
    loss_monotonic_sun_sem,
    cosine_divergence_I, entropy_I_attention, env_attention_penalty,
    self_attention_mass_from_attn, fstat_loss,
)
from metrics import semantic_health_metrics, epsilon_between_models, epsilon_to_alpha
from lr_by_epoch import LrByEpochSchedule

import csv
from pathlib import Path
import os
import math
import pandas as pd
from datetime import datetime
import random
import numpy as np


LOG_FIELDS = [
  "epoch","alpha","epsilon","dC","dM","d_cf","d_mono","d_att","d_self",
  "self_f","self_g","Val_sem","self_mass",
  "env_sum_tr","I_sum_tr","self_m_tr","corr_tr",
  "env_sum_val","I_sum_val","self_m_val","corr_val",
  "alpha_used",
  "y_stat_mean","y_sem_mean","y_mix_mean",
  "corr_stat","corr_sem","corr_mix",
  "cf_s0_stat_mean","cf_s0_sem_mean","cf_s0_mix_mean",
  "cf_p0_stat_mean","cf_p0_sem_mean","cf_p0_mix_mean",
  "contrib_stat","contrib_sem",
  "eps_mode","eps_override","alpha_override",
]

def append_row_csv(path, fieldnames, row_dict):
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        # ★余計なキーは捨てる（ここが重要）
        safe = {k: row_dict.get(k, "") for k in fieldnames}
        w.writerow(safe)

def init_alpha_log(path: Path, fieldnames, resume: bool):
    """
    新規run: alpha_log.csv を必ず作り直してヘッダを書く
    resume : 触らない（追記継続）
    """
    if resume:
        return
    # 新規化：既存があっても上書き
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

def read_last_logged_epoch_and_alpha(log_path: Path):
    """
    resume時に alpha_log.csv の最終epochとalpha(=次epoch向けalpha)を読む。
    無ければ (None, None)
    """
    if not log_path.exists():
        return None, None
    try:
        df = pd.read_csv(log_path)
        if len(df) == 0 or ("epoch" not in df.columns):
            return None, None
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.dropna(subset=["epoch"]).sort_values("epoch")
        last = df.iloc[-1]
        last_epoch = int(last["epoch"])
        last_alpha = None
        if "alpha" in df.columns:
            last_alpha = pd.to_numeric(last["alpha"], errors="coerce")
            if pd.isna(last_alpha):
                last_alpha = None
            else:
                last_alpha = float(last_alpha)
        return last_epoch, last_alpha
    except Exception:
        return None, None

def seed_all(seed: int):
    import os, random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# -----------------------------
# extra logging helpers (minimal)
# -----------------------------
_EPS = 1e-8
@torch.no_grad()
def _pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.view(-1)
    b = b.view(-1)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (a0.std(unbiased=False) * b0.std(unbiased=False) + eps)
    return float(((a0 * b0).mean() / denom).item())

@torch.no_grad()
def branch_log_dict(model: MultiIWorldModel, data, alpha: float, device) -> dict:
    """
    y_stat / y_sem のログ（反事実・相関・寄与率）をまとめて返す
    """
    p, s, w, y = data

    y_stat, y_sem, _, _ = model(p, s, w, return_both=True)
    y_mix = (1.0 - alpha) * y_stat + alpha * y_sem
    # counterfactual (sun=0 / plant=0)
    s0 = torch.zeros_like(s)
    p0 = torch.zeros_like(p)
    y_stat_s0, y_sem_s0, _, _ = model(p, s0, w, return_both=True)
    y_stat_p0, y_sem_p0, _, _ = model(p0, s, w, return_both=True)
    y_mix_s0 = (1.0 - alpha) * y_stat_s0 + alpha * y_sem_s0
    y_mix_p0 = (1.0 - alpha) * y_stat_p0 + alpha * y_sem_p0

    # correlations (real)
    corr_stat = _pearson_corr(y_stat, y)
    corr_sem  = _pearson_corr(y_sem, y)
    corr_mix  = _pearson_corr(y_mix, y)

    # contribution ratio (mean abs contribution)
    c_stat = ((1.0 - alpha) * y_stat).abs().mean()
    c_sem  = (alpha * y_sem).abs().mean()
    contrib_sem = float((c_sem / (c_stat + c_sem + _EPS)).item())
    contrib_stat = 1.0 - contrib_sem

    return {
        "y_stat_mean": float(y_stat.mean().item()),
        "y_sem_mean":  float(y_sem.mean().item()),
        "y_mix_mean":  float(y_mix.mean().item()),
        "corr_stat": corr_stat,
        "corr_sem":  corr_sem,
        "corr_mix":  corr_mix,
        "cf_s0_stat_mean": float(y_stat_s0.mean().item()),
        "cf_s0_sem_mean":  float(y_sem_s0.mean().item()),
        "cf_s0_mix_mean":  float(y_mix_s0.mean().item()),
        "cf_p0_stat_mean": float(y_stat_p0.mean().item()),
        "cf_p0_sem_mean":  float(y_sem_p0.mean().item()),
        "cf_p0_mix_mean":  float(y_mix_p0.mean().item()),
        "contrib_stat": contrib_stat,
        "contrib_sem":  contrib_sem,
    }

# =============================
# Experiment 3: "Island" logging
# =============================

# でかいログを避けるため、固定サブサンプルだけ保存（必要なら増やす）
ISLAND_MAX_SAMPLES = int(os.environ.get("ISLAND_MAX_SAMPLES", "512"))
# epoch % LOG_EVERY == 0 のタイミングで保存（必要なら環境変数で上書き）
ISLAND_EVERY = int(os.environ.get("ISLAND_EVERY", str(C.LOG_EVERY)))

@torch.no_grad()
def save_island_snapshot(
    model: MultiIWorldModel,
    data,
    epoch: int,
    alpha_used: float,
    device: str,
    out_dir: Path = "",
    max_samples: int = ISLAND_MAX_SAMPLES,
) -> Path:
    """
    「島」可視化用ログ（I埋め込み + growth行attention + 予測/反事実）を
    epochごとに1ファイルへ保存する。

    保存形式: torch.save(dict) -> runs/islands/island_epochXXXXX.pt
    """
    p, s, w, y = data
    B = p.size(0)
    n = min(B, max_samples)

    # 固定サブサンプル（先頭n）: まずは最小差分。
    # 「ラン間再現性」を取りたいなら、固定seedでランダムindexを作って保存する方式に後で拡張。
    p0 = p[:n]
    s0 = s[:n]
    w0 = w[:n]
    y0 = y[:n]

    # forward（stat/sem + attn + I）
    y_stat, y_sem, attn, (I1, I2, I3) = model(p0, s0, w0, return_both=True)
    y_mix = (1.0 - alpha_used) * y_stat + alpha_used * y_sem

    # counterfactual（sun=0 / plant=0）: クラスタごとの差を見るために一緒に保存
    z_s = torch.zeros_like(s0)
    z_p = torch.zeros_like(p0)
    y_stat_sun0, y_sem_sun0, _, _ = model(p0, z_s, w0, return_both=True)
    y_stat_plt0, y_sem_plt0, _, _ = model(z_p, s0, w0, return_both=True)
    y_mix_sun0 = (1.0 - alpha_used) * y_stat_sun0 + alpha_used * y_sem_sun0
    y_mix_plt0 = (1.0 - alpha_used) * y_stat_plt0 + alpha_used * y_sem_plt0

    # attention: growth query 行（head平均）→ (n, 7)
    # attn: (B,H,7,7) なので head平均して行6を抜く
    attn_grow = attn.mean(dim=1)[:, 6, :]  # (n,7)

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "alpha_used": float(alpha_used),
        "env": torch.stack([p0, s0, w0], dim=-1).detach().cpu(),      # (n,3)
        "y_true": y0.detach().cpu(),                                  # (n,)
        "y_stat": y_stat.detach().cpu(),
        "y_sem": y_sem.detach().cpu(),
        "y_mix": y_mix.detach().cpu(),
        "y_sun0_stat": y_stat_sun0.detach().cpu(),
        "y_sun0_sem": y_sem_sun0.detach().cpu(),
        "y_sun0_mix": y_mix_sun0.detach().cpu(),
        "y_plant0_stat": y_stat_plt0.detach().cpu(),
        "y_plant0_sem": y_sem_plt0.detach().cpu(),
        "y_plant0_mix": y_mix_plt0.detach().cpu(),
        "I": torch.cat([I1, I2, I3], dim=1).squeeze(2).detach().cpu()  # (n,3,D)
             if I1.dim() == 3 else torch.stack([I1, I2, I3], dim=1).detach().cpu(),
        "attn_growth_row": attn_grow.detach().cpu(),                  # (n,7)
        # ★追加：island_eps.py が探す個別データキー
        "plant": p0.detach().cpu(),
        "sun":   s0.detach().cpu(),
        "water": w0.detach().cpu(),
        "growth": y0.detach().cpu(),

        # ★追加：island_eps.py が探す attn キー（既存の attn_growth_row の別名）
        "attn": attn_grow.detach().cpu(),

        # ★追加：island_eps.py が探す model_state
        "model_state": copy.deepcopy(model.state_dict()),
    }

    out_path = out_dir / f"island_epoch{epoch:05d}.pt"
    torch.save(payload, out_path)
    return out_path

def main():
    print("device:", C.device)
    
    # ---- SEED ----
    SEED = int(os.environ.get("SEED", "0"))
    print(f"[SEED] SEED={SEED}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 可能な範囲で再現性を上げる（少し遅くなる場合あり）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = MultiIWorldModel(d_model=C.D_MODEL, n_heads=C.N_HEADS).to(C.device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)

    # ---- quick determinism check (data) ----
    seed_all(SEED)
    plant1, sun1, water1, growth1 = generate_world_data(16)

    seed_all(SEED)
    plant2, sun2, water2, growth2 = generate_world_data(16)

    print("[CHK] plant equal:", torch.equal(plant1, plant2))
    print("[CHK] sun   equal:", torch.equal(sun1, sun2))
    print("[CHK] water equal:", torch.equal(water1, water2))
    print("[CHK] growth equal:", torch.equal(growth1, growth2))

    # ---- RUN_ID / RUN_DIR ----
    # 1) 環境変数 RUN_ID があればそれを使う
    # 2) なければ時刻で自動採番（衝突しにくい）
    RUN_ID = os.environ.get("RUN_ID", "").strip()
    if RUN_ID == "":
        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    RUN_DIR = Path("runs") / RUN_ID
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] RUN_ID={RUN_ID}")
    print(f"[RUN] RUN_DIR={RUN_DIR}")

    # ---- epsilon/alpha intervention (minimal) ----
    EPS_MODE = os.environ.get("EPS_MODE", "normal").strip().lower()
    # EPS_MODE: "normal" | "freeze" | "override"
    EPS_OVERRIDE = os.environ.get("EPS_OVERRIDE", "").strip()
    ALPHA_OVERRIDE = os.environ.get("ALPHA_OVERRIDE", "").strip()

    eps_override_val = None
    if EPS_OVERRIDE != "":
        try:
            eps_override_val = float(EPS_OVERRIDE)
        except Exception:
            raise ValueError(f"Invalid EPS_OVERRIDE={EPS_OVERRIDE}")

    alpha_override_val = None
    if ALPHA_OVERRIDE != "":
        try:
            alpha_override_val = float(ALPHA_OVERRIDE)
        except Exception:
            raise ValueError(f"Invalid ALPHA_OVERRIDE={ALPHA_OVERRIDE}")

    print(f"[EPS] mode={EPS_MODE} EPS_OVERRIDE={eps_override_val} ALPHA_OVERRIDE={alpha_override_val}")

    # ---- Output paths (RUN_DIR based) ----
    LOG_PATH = RUN_DIR / "alpha_log.csv"

    fstat = FStatWrapper(model, device=C.device, tau=C.EMA_TAU, scale=C.FSTAT_SCALE)
    prev_state = None
    alpha = 0.5

    # If user forces alpha from the beginning
    if alpha_override_val is not None:
        alpha = float(alpha_override_val)

        # ---- RESUME policy ----
    RESUME = os.environ.get("RESUME", "0").strip() == "1"

    # 新規runは必ず alpha_log を作り直す（追記事故防止）
    init_alpha_log(LOG_PATH, LOG_FIELDS, resume=RESUME)

    # resume時：alpha_log の最終epochを見て start_epoch を決める
    start_epoch = 0
    if RESUME:
        last_epoch, last_alpha = read_last_logged_epoch_and_alpha(LOG_PATH)
        if last_epoch is not None:
            start_epoch = last_epoch + 1
        # alpha も引き継ぎたいなら（無ければ 0.5 のまま）
        if last_alpha is not None:
            alpha = float(last_alpha)  # ※alpha変数をこの時点で定義する必要がある（後述）

    ISLAND_DIR = RUN_DIR / "islands"
    ISLAND_DIR.mkdir(parents=True, exist_ok=True)

    # もし derived も train から出したいなら、ここで作っておく（任意）
    DERIVED_DIR = RUN_DIR / "derived"
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    plant, sun, water, growth = generate_world_data(C.N)
    train_idx, val_idx = make_split(C.N, C.TRAIN_RATIO, seed=SEED)
    train_data = pack_data(plant, sun, water, growth, train_idx, C.device)
    val_data   = pack_data(plant, sun, water, growth, val_idx, C.device)

    # ★追加：クラスタ条件付きLR（外部CSVでepoch→lr_multを引く）
    # 例: runs/run1/lr_mult_by_epoch.csv を手で用意しておく（解析スクリプトの出力）
    DEFAULT_LR_TABLE = str(RUN_DIR / "lr_mult_by_epoch.csv")
    LR_TABLE = os.environ.get("LR_TABLE", DEFAULT_LR_TABLE)
    
    def load_lr_table_series(path: str):
        if (path is None) or (path == ""):
            return None
        if not os.path.exists(path):
            print(f"[LR] table not found: {path} (disable)")
            return None

        df = pd.read_csv(path)
        if "epoch" not in df.columns:
            raise ValueError(f"LR table missing 'epoch' column: {df.columns.tolist()}")
        if "lr_mult" not in df.columns:
            raise ValueError(f"LR table missing 'lr_mult' column: {df.columns.tolist()}")

        df["epoch"] = df["epoch"].astype(int)
        df["lr_mult"] = pd.to_numeric(df["lr_mult"], errors="coerce")

        # epoch昇順
        df = df.sort_values("epoch")

        # ★追加：epoch 重複は「最後の行」を採用
        df = df.drop_duplicates(subset=["epoch"], keep="last")

        # lr_mult 欠損は落とす
        df = df.dropna(subset=["lr_mult"])

        s = df.set_index("epoch")["lr_mult"]
        return s

    lr_series = load_lr_table_series(LR_TABLE)

    if lr_series is not None and len(lr_series) > 0:
        print(f"[LR] using lr schedule table: {LR_TABLE}")
        print("[LR] table preview (head):")
        print(lr_series.head().to_string())
        print("[LR] table preview (tail):")
        print(lr_series.tail().to_string())
    else:
        print("[LR] disabled (no valid lr table)")

    def safe_lr_mult(epoch: int) -> float:
        if lr_series is None or len(lr_series) == 0:
            return 1.0

        # forward-fill：epoch以下で最大のキーを探す
        # （epoch=201なら200の値を使う）
        idx = lr_series.index.values
        # searchsortedで「挿入位置」を求め、1つ戻す
        pos = idx.searchsorted(epoch, side="right") - 1
        if pos < 0:
            mult = 1.0
        else:
            mult = float(lr_series.iloc[pos])

        if not math.isfinite(mult):
            mult = 1.0

        # 安全クリップ
        mult = max(0.1, min(mult, 2.0))
        return mult

    for epoch in range(start_epoch, C.EPOCHS):
        mult = safe_lr_mult(epoch)
        lr_now = C.LR * mult
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        if epoch % C.LOG_EVERY == 0:
            print(f"[LR] epoch={epoch} mult={mult:.4f} lr={lr_now:.6g}")

        model.train()
        opt.zero_grad()

        # α から λ_eff
        λ_cf_eff   = C.LAMBDA["cf_base"]   * (0.5 + 0.5 * alpha)
        λ_mono_eff = C.LAMBDA["mono_base"] * (0.5 + 0.5 * alpha)
        λ_cos_eff  = C.LAMBDA["cos_base"]  * (0.5 + 0.5 * alpha)
        λ_ent_eff  = C.LAMBDA["ent_base"]  * (0.5 + 0.5 * alpha)
        λ_env_eff  = C.LAMBDA["env_base"]  * (0.5 + 0.5 * alpha)
        λ_real_eff = C.LAMBDA["real_base"] * (1.5 - 0.5 * alpha)

        # real (F2 mix)
        L_real_mix, attn, I_triplet = loss_real_with_attn_F2(model, train_data, alpha)

        # semantic weak fit (log-MSE)
        L_real_sem = loss_real_sem_log(model, train_data)

        # constraints (semantic only)
        L_cf_s = loss_counterfactual_sun0_sem(model, train_data)
        L_cf_p = loss_counterfactual_plant0_sem(model, train_data)
        L_mono = loss_monotonic_sun_sem(model, C.device, n_pairs=128)

        L_cos = cosine_divergence_I(I_triplet)
        H_I   = entropy_I_attention(attn)
        L_env = env_attention_penalty(attn, alpha=0.3)

        # reverse head (A案: margin below -> push worse)
        p, s, w = train_data[:3]
        reverse_pred = model.reverse_from_growth(batch_size=p.size(0), device=C.device)
        env_do_s = torch.stack([p, torch.zeros_like(s), w], dim=-1)
        env_do_p = torch.stack([torch.zeros_like(p), s, w], dim=-1)
        mse_s = F.mse_loss(reverse_pred, env_do_s)
        mse_p = F.mse_loss(reverse_pred, env_do_p)
        mse = 0.5 * (mse_s + mse_p)
        L_reverse = torch.relu(C.REV_MARGIN - mse)

        rev_abs_mean = reverse_pred.abs().mean()
        rev_std = reverse_pred.std()

        # self-loop regularization
        self_mass = self_attention_mass_from_attn(attn)
        L_self = (self_mass - C.SELF_TARGET).abs()

        λ_sem_eff = max(C.LAMBDA_SEM_MIN, C.LAMBDA["sem"] * alpha)

        loss = (
            λ_real_eff * L_real_mix
            + λ_sem_eff * L_real_sem
            + λ_cf_eff   * (L_cf_s + L_cf_p)
            + λ_mono_eff * L_mono
            + λ_cos_eff  * L_cos
            - λ_ent_eff  * H_I
            + λ_env_eff  * L_env
            + C.LAMBDA["self_"] * L_self
            + C.LAMBDA["reverse"] * L_reverse
        )

        loss.backward()
        opt.step()

        fstat.ema_update(model)
        _ = fstat_loss(fstat, train_data, device=C.device, base_loss_weight=0.1, meaning_weight=0.05)

        if epoch % C.LOG_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_L_sem = loss_real_sem_log(model, val_data)
                mon_tr  = semantic_health_metrics(model, train_data, C.device)
                mon_val = semantic_health_metrics(model, val_data, C.device)
                # NEW: y_stat/y_sem logs (uses current alpha)
                alpha_used = float(alpha)
                br = branch_log_dict(model, train_data, alpha=alpha_used, device=C.device)
            
            # Experiment 3: island snapshot（I埋め込み + attention行）
            if epoch % ISLAND_EVERY == 0:
                path = save_island_snapshot(
                    model=model,
                    data=train_data,
                    epoch=epoch,
                    alpha_used=alpha_used,
                    device=C.device,
                    out_dir=ISLAND_DIR,
                )
                print(f"   [ISLAND] saved snapshot: {path}")

            print(
                f"[{epoch}] "
                f"L_real_mix={L_real_mix.item():.4f} "
                f"L_real_sem={L_real_sem.item():.4f} "
                f"L_cf_s0={L_cf_s.item():.4f} L_cf_p0={L_cf_p.item():.4f} "
                f"L_mono={L_mono.item():.4f} "
                f"L_cos={L_cos.item():.4f} "
                f"H_I={H_I.item():.4f} "
                f"L_env={L_env.item():.4f} "
                f"L_self={L_self.item():.4f} "
                f"L_rev={L_reverse.item():.4f} "
                f"mse_s={mse_s.item():.4f} mse_p={mse_p.item():.4f} mse={mse.item():.4f} "
                f"rev_abs_mean={rev_abs_mean.item():.4f} rev_std={rev_std.item():.4f} "
                f"self_mass={self_mass.item():.4f} "
                f"Val_sem={val_L_sem.item():.4f}"
            )

            print(
                f"   [MON-TR] y_mean={mon_tr['y_mean']:.4f} y_std={mon_tr['y_std']:.4f} "
                f"y_min={mon_tr['y_min']:.4f} y_max={mon_tr['y_max']:.4f} "
                f"corr={mon_tr['corr']:.4f} "
                f"m_real={mon_tr['m_real']:.4f} m_s0={mon_tr['m_s0']:.4f} m_p0={mon_tr['m_p0']:.4f} "
                f"gap_s={mon_tr['gap_s']:.4f} gap_p={mon_tr['gap_p']:.4f} "
                f"ratio_s={mon_tr['ratio_s']:.4f} ratio_p={mon_tr['ratio_p']:.4f} "
                f"env_sum={mon_tr['env_sum']:.4f} I_sum={mon_tr['I_sum']:.4f} self={mon_tr['self_m']:.4f} "
                f"H_I={mon_tr['H_I']:.4f}"
            )
            print(
                f"   [MON-VA] y_mean={mon_val['y_mean']:.4f} y_std={mon_val['y_std']:.4f} "
                f"y_min={mon_val['y_min']:.4f} y_max={mon_val['y_max']:.4f} "
                f"corr={mon_val['corr']:.4f} "
                f"m_real={mon_val['m_real']:.4f} m_s0={mon_val['m_s0']:.4f} m_p0={mon_val['m_p0']:.4f} "
                f"gap_s={mon_val['gap_s']:.4f} gap_p={mon_val['gap_p']:.4f} "
                f"ratio_s={mon_val['ratio_s']:.4f} ratio_p={mon_val['ratio_p']:.4f} "
                f"env_sum={mon_val['env_sum']:.4f} I_sum={mon_val['I_sum']:.4f} self={mon_val['self_m']:.4f} "
                f"H_I={mon_val['H_I']:.4f}"
            )

            if prev_state is not None:
                prev_model = MultiIWorldModel(d_model=C.D_MODEL, n_heads=C.N_HEADS).to(C.device)
                prev_model.load_state_dict(prev_state)
                prev_model.eval()

                eps_info = epsilon_between_models(
                    model_f=model, model_g=prev_model, data=train_data, device=C.device,
                    w_cf=1.0, w_mono=1.0, w_att=1.0, w_self=1.0, sym_att=True,
                )
                epsilon_val = float(eps_info["epsilon"])

                # ---- intervention: decide alpha_next ----
                alpha_prev = float(alpha)  # current alpha before update

                if alpha_override_val is not None:
                    # Strongest: alpha is fixed externally
                    alpha_next = float(alpha_override_val)
                    mode_used = "alpha_override"
                elif EPS_MODE == "freeze":
                    # Freeze: do not update alpha by epsilon gate
                    alpha_next = alpha_prev
                    mode_used = "freeze"
                elif EPS_MODE == "override":
                    # Override epsilon value used for alpha update (still uses epsilon_to_alpha)
                    if eps_override_val is None:
                        raise ValueError("EPS_MODE=override requires EPS_OVERRIDE to be set.")
                    alpha_next = float(epsilon_to_alpha(float(eps_override_val), k=C.ALPHA_K))
                    mode_used = f"eps_override({eps_override_val})"
                else:
                    # normal behavior
                    alpha_next = float(epsilon_to_alpha(epsilon_val, k=C.ALPHA_K))
                    mode_used = "normal"

                alpha = alpha_next

                row = {
                    "epoch": epoch,
                    # alpha used for the logged forward (this epoch)
                    "alpha_used": alpha_used,
                    # alpha updated by epsilon gate (for next epochs)
                    "alpha": alpha_next,
                    "epsilon": epsilon_val,
                    "dC": eps_info["dC"],
                    "dM": eps_info["dM"],
                    "d_cf": eps_info["d_cf"],
                    "d_mono": eps_info["d_mono"],
                    "d_att": eps_info["d_att"],
                    "d_self": eps_info["d_self"],
                    "self_f": eps_info["self_f"],
                    "self_g": eps_info["self_g"],
                    "Val_sem": float(val_L_sem.item()),
                    "self_mass": float(self_mass.item()),
                    # あると便利（semantic_health_metrics の結果）
                    "env_sum_tr": float(mon_tr["env_sum"]),
                    "I_sum_tr": float(mon_tr["I_sum"]),
                    "self_m_tr": float(mon_tr["self_m"]),
                    "corr_tr": float(mon_tr["corr"]),
                    "env_sum_val": float(mon_val["env_sum"]),
                    "I_sum_val": float(mon_val["I_sum"]),
                    "self_m_val": float(mon_val["self_m"]),
                    "corr_val": float(mon_val["corr"]),
                    "eps_mode": mode_used,
                    "eps_override": ("" if eps_override_val is None else float(eps_override_val)),
                    "alpha_override": ("" if alpha_override_val is None else float(alpha_override_val)),
                    # NEW: branch logs
                    **br,
                }
                for k in LOG_FIELDS:
                    row.setdefault(k, "")
                if RESUME:
                    last_epoch, _ = read_last_logged_epoch_and_alpha(LOG_PATH)
                    if (last_epoch is not None) and (epoch <= last_epoch):
                        print(f"[LOG] skip duplicate epoch={epoch} (last_epoch_in_log={last_epoch})")
                    else:
                        append_row_csv(LOG_PATH, LOG_FIELDS, row)
                else:
                    append_row_csv(LOG_PATH, LOG_FIELDS, row)
                
                print(
                    f"   dC={eps_info['dC']:.6f} "
                    f"d_cf={eps_info['d_cf']:.6f} "
                    f"d_mono={eps_info['d_mono']:.6f} "
                    f"d_att={eps_info['d_att']:.6f} "
                    f"d_self={eps_info['d_self']:.6f} "
                    f"self_f={eps_info['self_f']:.4f} "
                    f"self_g={eps_info['self_g']:.4f} "
                    f"epsilon={epsilon_val:.6f} "
                    f"alpha={alpha:.3f}"
                )

            prev_state = copy.deepcopy(model.state_dict())

    # 便利：最後に重み保存したいなら
    torch.save(model.state_dict(), RUN_DIR / "model_last.pt")

if __name__ == "__main__":
    main()
