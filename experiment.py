import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# -----------------------------
# 1. データ生成（前回と同じ）
# -----------------------------
def generate_world_data(n):
    plant = torch.bernoulli(torch.full((n,), 0.7))
    sun   = torch.rand(n)
    water = torch.rand(n)

    base = 1.2*sun + 0.8*water
    base = torch.tanh(base * 2.0)
    noise = 0.05*torch.randn(n)
    growth = plant * (base + noise)
    growth = torch.clamp(growth, min=0.0)
    return plant, sun, water, growth

N = 5000
plant, sun, water, growth = generate_world_data(N)

perm = torch.randperm(N)
train_idx = perm[: int(0.8*N)]
val_idx   = perm[int(0.8*N):]

train_data = (
    plant[train_idx].to(device),
    sun[train_idx].to(device),
    water[train_idx].to(device),
    growth[train_idx].to(device),
)
val_data = (
    plant[val_idx].to(device),
    sun[val_idx].to(device),
    water[val_idx].to(device),
    growth[val_idx].to(device),
)

# -----------------------------
# 2. Multi-head Self-Attention
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None, need_weights=False):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)

        if attn_mask is not None:
            if attn_mask.shape != scores.shape:
                raise ValueError(f"attn_mask shape {attn_mask.shape} must match scores {scores.shape}")
            scores = attn_mask.to(dtype=scores.dtype, device=scores.device).add_(scores)

        attn = torch.softmax(scores, dim=-1)           # (B,H,T,T)
        out = torch.matmul(attn, v)                    # (B,H,T,Dh)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)

        if need_weights:
            return out, attn
        return out, None

# -----------------------------
# 3. Multi-I 世界モデル (F2 用拡張)
#    トークン列: [plant, sun, water, I1, I2, I3, growth]
# -----------------------------
class MultiIWorldModel(nn.Module):
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        self.d_model = d_model

        # 外界トークン
        self.embed_in = nn.Linear(1, d_model)

        # 3つの I を作る MLP
        self.I_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 3 * d_model),  # I1,I2,I3 をまとめて出す
        )

        # growth クエリトークン
        self.growth_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)

        # 意味ブランチ用 head（従来の growth token から）
        self.head_sem = nn.Linear(d_model, 1)
        # 統計ブランチ用 head（env 平均から）
        self.head_stat = nn.Linear(d_model, 1)
        # 逆向き head（growth token から env を推定）
        self.head_reverse = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 3),
        )

    def forward(self, plant, sun, water,
                return_attn=False, return_both=False):
        B = plant.shape[0]

        p = self.embed_in(plant.view(B, 1, 1))
        s = self.embed_in(sun.view(B, 1, 1))
        w = self.embed_in(water.view(B, 1, 1))

        env = torch.stack([plant, sun, water], dim=-1)      # (B,3)
        I_all = self.I_mlp(env)                             # (B,3D)
        I_all = I_all.view(B, 3, self.d_model)              # (B,3,D)
        I1, I2, I3 = I_all[:, 0:1, :], I_all[:, 1:2, :], I_all[:, 2:3, :]

        g = self.growth_token.expand(B, -1, -1)             # (B,1,D)

        # [plant, sun, water, I1, I2, I3, growth]
        x = torch.cat([p, s, w, I1, I2, I3, g], dim=1)      # (B,7,D)

        # Attention mask (下三角 + I 限定参照):
        #   - env 同士は時間方向の下三角（未来禁止）
        #   - I1-3 は env(plant,sun,water) のみ参照可
        #   - growth は過去/現在のみ参照可（下三角）
        T = x.size(1)
        base = torch.zeros((T, T), device=x.device, dtype=x.dtype)
        upper = torch.triu(torch.ones((T, T), device=x.device), diagonal=1).bool()
        base = base.masked_fill(upper, float("-inf"))
        base[3:6, :] = float("-inf")
        base[3:6, 0:3] = 0.0

        attn_mask = base.unsqueeze(0).unsqueeze(0).expand(B, self.attn.n_heads, T, T).clone()

        x_out, attn = self.attn(x, attn_mask=attn_mask, need_weights=True)       # attn:(B,H,7,7)

        # semantic / meaning ブランチ
        growth_sem_rep  = x_out[:, 6, :]                    # growth token
        growth_sem_raw  = self.head_sem(growth_sem_rep).squeeze(-1)
        growth_sem_pred = F.softplus(growth_sem_raw)

        # statistical ブランチ
        env_reps  = x_out[:, 0:3, :]                        # plant,sun,water
        env_mean  = env_reps.mean(dim=1)                    # (B,D)
        growth_stat_raw  = self.head_stat(env_mean).squeeze(-1)
        growth_stat_pred = F.softplus(growth_stat_raw)

        if return_both:
            return growth_stat_pred, growth_sem_pred, attn, (I1, I2, I3)

        if return_attn:
            return growth_sem_pred, attn, (I1, I2, I3)

        # デフォルトは semantic 出力
        return growth_sem_pred

    def reverse_from_growth(self, batch_size, device=None):
        """
        growth トークンのみを残し、他トークンをゼロ埋めした入力から
        env (plant, sun, water) を逆推定するヘッド。
        """
        if device is None:
            device = self.growth_token.device

        g = self.growth_token.expand(batch_size, -1, -1).to(device=device)
        zeros_env = torch.zeros(batch_size, 6, self.d_model, device=device, dtype=g.dtype)
        x_rev = torch.cat([zeros_env, g], dim=1)

        x_rev_out, _ = self.attn(x_rev, need_weights=False)
        growth_rev_rep = x_rev_out[:, 6, :]
        reverse_raw = self.head_reverse(growth_rev_rep)
        return reverse_raw

class FStatWrapper(nn.Module):
    """
    main_model の EMA 版（teacher）
    出力にはスケール制御を入れる。
    """
    def __init__(self, model, tau=0.02, scale=2.0):
        super().__init__()
        self.teacher = MultiIWorldModel(d_model=model.d_model,
                                        n_heads=model.attn.n_heads).to(device)
        self.teacher.load_state_dict(model.state_dict())   # 初期は同じ
        self.tau = tau
        self.scale = scale

    @torch.no_grad()
    def ema_update(self, model):
        for p_teacher, p_main in zip(self.teacher.parameters(), model.parameters()):
            p_teacher.data = (1 - self.tau) * p_teacher.data + self.tau * p_main.data

    def forward(self, plant, sun, water):
        raw = self.teacher(plant, sun, water)
        y = torch.tanh(raw) * self.scale
        y = F.softplus(y)
        return y

# -----------------------------
# 4. loss 関数群
# -----------------------------
def loss_real_with_attn_F2(model, data, alpha):
    """
    F2: y = (1-alpha) * y_stat + alpha * y_sem で予測
    """
    p, s, w, y = data
    y_stat, y_sem, attn, I_triplet = model(p, s, w, return_both=True)
    y_pred = (1.0 - alpha) * y_stat + alpha * y_sem
    mse = F.mse_loss(y_pred, y)
    return mse, attn, I_triplet

def loss_counterfactual_sun0(model, data):
    p, s, w, _ = data
    s0 = torch.zeros_like(s)
    pred = model(p, s0, w)
    return (pred ** 2).mean()

def loss_counterfactual_plant0(model, data):
    p, s, w, _ = data
    p0 = torch.zeros_like(p)
    pred = model(p0, s, w)
    return (pred ** 2).mean()

def loss_monotonic_sun(model, device, n_pairs=64):
    plant = torch.ones(n_pairs, device=device)
    water = torch.rand(n_pairs, device=device)
    sun1  = torch.rand(n_pairs, device=device)
    sun2  = sun1 + torch.rand(n_pairs, device=device) * (1.0 - sun1)

    y1 = model(plant, sun1, water)
    y2 = model(plant, sun2, water)
    return torch.relu(y1 - y2).mean()

def cosine_divergence_I(I_triplet):
    I1, I2, I3 = I_triplet
    def cos(a, b):
        a_flat = a.view(a.size(0), -1)
        b_flat = b.view(b.size(0), -1)
        return F.cosine_similarity(a_flat, b_flat, dim=-1)  # (B,)
    c12 = cos(I1, I2)
    c13 = cos(I1, I3)
    c23 = cos(I2, I3)
    return (c12**2 + c13**2 + c23**2).mean()

def entropy_I_attention(attn):
    B, H, T, _ = attn.shape
    attn_mean = attn.mean(dim=1)          # (B,7,7)
    growth_row = attn_mean[:, 6, :]       # (B,7)
    probs_I = growth_row[:, 3:6]          # I1,I2,I3

    eps = 1e-8
    log_p = torch.log(probs_I + eps)
    ent = -(probs_I * log_p).sum(dim=-1)  # (B,)
    return ent.mean()

def env_attention_penalty(attn, alpha=0.3):
    B, H, T, _ = attn.shape
    attn_mean = attn.mean(dim=1)          # (B,7,7)
    growth_row = attn_mean[:, 6, :]       # (B,7)
    env_sum = growth_row[:, 0:3].sum(dim=-1)
    penalty = torch.relu(alpha - env_sum)
    return penalty.mean()

# === ここから self 関連のヘルパー ===

def self_attention_mass_from_attn(attn):
    """
    growth クエリが growth_self へ向ける注意の平均を返す。
    attn: (B,H,7,7)
    """
    B, H, T, _ = attn.shape
    attn_mean = attn.mean(dim=1)          # (B,7,7)
    growth_row = attn_mean[:, 6, :]       # (B,7)
    self_mass = growth_row[:, 6]          # (B,)
    return self_mass.mean()

def fstat_loss(fstat, data, base_loss_weight=0.1, meaning_weight=0.05):
    p, s, w, y = data
    y_pred = fstat(plant=p, sun=s, water=w)
    L_basic = (y_pred ** 2).mean()

    # weak causal / monotonic constraints
    y_s0 = fstat(p, torch.zeros_like(s), w)
    y_p0 = fstat(torch.zeros_like(p), s, w)
    L_cf = (y_s0.mean() + y_p0.mean())

    plant = torch.ones(64, device=device)
    water = torch.rand(64, device=device)
    sun1  = torch.rand(64, device=device)
    sun2  = sun1 + torch.rand(64, device=device) * (1.0 - sun1)
    y1 = fstat(plant, sun1, water)
    y2 = fstat(plant, sun2, water)
    L_mono = torch.relu(y1 - y2).mean()

    return (
        base_loss_weight * L_basic
        + meaning_weight * (L_cf + L_mono)
    )

# =========================================
# semantic ブランチ専用の因果・単調性制約
# =========================================

def loss_counterfactual_sun0_sem(model, data):
    """
    semantic ブランチにだけ sun=0 反事実制約を課す。
    F2 の混合出力ではなく、y_sem 単独にペナルティ。
    """
    p, s, w, _ = data
    s0 = torch.zeros_like(s)

    # F2 用 forward: return_both=True で (y_stat, y_sem, attn, I_triplet)
    y_stat, y_sem, _, _ = model(p, s0, w, return_both=True)

    # 「sun=0 なら成長してはいけない」は semantic 側にだけ要求
    return (y_sem ** 2).mean()


def loss_counterfactual_plant0_sem(model, data):
    """
    semantic ブランチにだけ plant=0 反事実制約を課す。
    """
    p, s, w, _ = data
    p0 = torch.zeros_like(p)

    y_stat, y_sem, _, _ = model(p0, s, w, return_both=True)
    return (y_sem ** 2).mean()


def loss_monotonic_sun_sem(model, device, n_pairs=64):
    """
    semantic ブランチにだけ sun 単調性制約を課す。
    plant=1, water ランダム、sun1 <= sun2 なのに y_sem(sun1) > y_sem(sun2)
    になっている部分を罰する。
    """
    plant = torch.ones(n_pairs, device=device)
    water = torch.rand(n_pairs, device=device)
    sun1  = torch.rand(n_pairs, device=device)
    sun2  = sun1 + torch.rand(n_pairs, device=device) * (1.0 - sun1)  # sun2 >= sun1

    y_stat1, y_sem1, _, _ = model(plant, sun1, water, return_both=True)
    y_stat2, y_sem2, _, _ = model(plant, sun2, water, return_both=True)

    viol = torch.relu(y_sem1 - y_sem2).mean()
    return viol


import copy
import math

# =============================
# ε 計算用ヘルパー
# =============================

@torch.no_grad()
def dC(model_f, model_g, data):
    p, s, w, _ = data
    y_f = model_f(p, s, w)
    y_g = model_g(p, s, w)
    return (y_f - y_g).abs().mean().item()

@torch.no_grad()
def cf_violation(model, data):
    p, s, w, _ = data
    y_s0 = model(p, torch.zeros_like(s), w)
    y_p0 = model(torch.zeros_like(p), s, w)
    return (y_s0.mean() + y_p0.mean()).item()

@torch.no_grad()
def mono_violation(model, device, n_pairs=64):
    plant = torch.ones(n_pairs, device=device)
    water = torch.rand(n_pairs, device=device)
    sun1  = torch.rand(n_pairs, device=device)
    sun2  = sun1 + torch.rand(n_pairs, device=device) * (1.0 - sun1)
    y1 = model(plant, sun1, water)
    y2 = model(plant, sun2, water)
    viol = torch.relu(y1 - y2).mean()
    return viol.item()

@torch.no_grad()
def attention_distribution_with_self(model, data):
    """
    growth クエリ行の 7 要素全体を分布化して返す:
    [plant, sun, water, I1, I2, I3, growth_self]
    """
    p, s, w, _ = data
    y_pred, attn, _ = model(p, s, w, return_attn=True)  # (B,H,7,7)

    attn_mean = attn.mean(dim=1).mean(dim=0)  # (7,7)
    growth_row = attn_mean[6]                 # (7,)

    dist7 = torch.clamp(growth_row, min=1e-8)
    dist7 = dist7 / dist7.sum()
    return dist7.cpu()

def KL_divergence(p, q):
    eps = 1e-8
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return float((p * (p.log() - q.log())).sum().item())

@torch.no_grad()
def dM(model_f, model_g, data, device,
       w_cf=1.0, w_mono=1.0, w_att=1.0, w_self=1.0,
       sym_att=True):
    """
    意味圏の距離 d_M(F(f),F(g)):
      d_cf, d_mono, d_att( env+I ), d_self( self_mass の差 )
    """

    # 1. 反事実違反度の差
    CF_f = cf_violation(model_f, data)
    CF_g = cf_violation(model_g, data)
    d_cf = abs(CF_f - CF_g)

    # 2. 単調性違反度の差
    MONO_f = mono_violation(model_f, device=device, n_pairs=128)
    MONO_g = mono_violation(model_g, device=device, n_pairs=128)
    d_mono = abs(MONO_f - MONO_g)

    # 3. Attention 分布 + self_mass
    dist_f = attention_distribution_with_self(model_f, data)
    dist_g = attention_distribution_with_self(model_g, data)

    # env+I の 6次元で KL
    p = dist_f[:6]
    q = dist_g[:6]
    p = p / p.sum()
    q = q / q.sum()

    if sym_att:
        d_att = KL_divergence(p, q) + KL_divergence(q, p)
    else:
        d_att = KL_divergence(p, q)

    # self_mass の差
    self_f = float(dist_f[6].item())
    self_g = float(dist_g[6].item())
    d_self = abs(self_f - self_g)

    dM_val = w_cf * d_cf + w_mono * d_mono + w_att * d_att + w_self * d_self

    return {
        "d_cf": d_cf,
        "d_mono": d_mono,
        "d_att": d_att,
        "d_self": d_self,
        "self_f": self_f,
        "self_g": self_g,
        "dM": dM_val,
    }

@torch.no_grad()
def epsilon_between_models(model_f, model_g, data, device,
                           w_cf=1.0, w_mono=1.0, w_att=1.0, w_self=1.0,
                           sym_att=True):
    dC_val = dC(model_f, model_g, data)
    dM_dict = dM(model_f, model_g, data, device,
                 w_cf=w_cf, w_mono=w_mono, w_att=w_att, w_self=w_self,
                 sym_att=sym_att)
    eps = dM_dict["dM"] - dC_val
    out = {
        "dC": dC_val,
        **dM_dict,
        "epsilon": eps,
    }
    return out

# =============================
# ε -> α (gate) 変換ヘルパー
# =============================
def epsilon_to_alpha(epsilon, k=5.0):
    import math
    eps_clipped = max(min(epsilon, 1.0), -1.0)
    return 1.0 / (1.0 + math.exp(-k * eps_clipped))

# -----------------------------
# 5. 学習ループ（F2 + L_self）
# -----------------------------
model = MultiIWorldModel(d_model=32, n_heads=4).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
fstat = FStatWrapper(model, tau=0.02, scale=2.0)

λ_cf_base   = 1.0
λ_mono_base = 0.5
λ_cos_base  = 0.1
λ_ent_base  = 0.05
λ_env_base  = 0.5
λ_real_base = 1.0
λ_self      = 1.0   # self-loop 正則化の重み（αとは独立）
λ_sem       = 0.1  # ★ semantic ブランチ用の弱い MSE
λ_reverse   = 0.01  # growth→env 逆向き head への負の学習率（小さめ）

self_target = 0.03  # growth_self への注意の望ましい上限（要調整）

prev_state = None
alpha = 0.5

for epoch in range(2001):
    model.train()
    opt.zero_grad()

    # α から λ_eff を計算
    λ_cf_eff   = λ_cf_base   * (0.5 + 0.5 * alpha)
    λ_mono_eff = λ_mono_base * (0.5 + 0.5 * alpha)
    λ_cos_eff  = λ_cos_base  * (0.5 + 0.5 * alpha)
    λ_ent_eff  = λ_ent_base  * (0.5 + 0.5 * alpha)
    λ_env_eff  = λ_env_base  * (0.5 + 0.5 * alpha)
    λ_real_eff = λ_real_base * (1.5 - 0.5 * alpha)

        # -----------------------------------------
    # F2 バージョンの実データ損失（stat+sem の混合）
    # -----------------------------------------
    L_real_mix, attn, I_triplet = loss_real_with_attn_F2(model, train_data, alpha)

    # NEW: semantic ブランチ単独のデータフィット（弱い責任）
    # alpha=1.0 で y_pred = y_sem になることを利用
    L_real_sem, _, _ = loss_real_with_attn_F2(model, train_data, alpha=1.0)

    # 反事実・単調性・I 構造（semantic ブランチにのみ課す）
    L_cf_s = loss_counterfactual_sun0_sem(model, train_data)
    L_cf_p = loss_counterfactual_plant0_sem(model, train_data)
    L_mono = loss_monotonic_sun_sem(model, device, n_pairs=128)

    L_cos = cosine_divergence_I(I_triplet)
    H_I   = entropy_I_attention(attn)
    L_env = env_attention_penalty(attn, alpha=0.3)

    # growth token のみを残した入力から env を再構成する逆向き head。
    # 介入データ (do(sun=0), do(plant=0)) をターゲットにしつつ、
    # その精度を下げるように負の重みで学習。
    p, s, w = train_data[:3]
    # --- reverse head: growth から env(do) を再構成しようとする能力を「一定以上なら抑制」する（A案） ---
    reverse_pred = model.reverse_from_growth(batch_size=p.size(0), device=device)

    # 介入ターゲット
    env_do_s = torch.stack([p, torch.zeros_like(s), w], dim=-1)
    env_do_p = torch.stack([torch.zeros_like(p), s, w], dim=-1)

    # 監視用：各介入ターゲットへの MSE（reduction="mean"）
    mse_s = F.mse_loss(reverse_pred, env_do_s)
    mse_p = F.mse_loss(reverse_pred, env_do_p)
    mse = 0.5 * (mse_s + mse_p)

    # A案：MSE が margin 未満のときだけ「もっと外せ（= mse を増やせ）」という圧をかける
    # mse が margin を超えたら 0 になり、発散で稼げない
    rev_margin = 1.0  # まずは 0.5, 1.0, 2.0 で試すのがおすすめ
    L_reverse = torch.relu(rev_margin - mse)

    # 監視用：reverse_pred の発散検知
    rev_abs_mean = reverse_pred.abs().mean()
    rev_std = reverse_pred.std()

    # self-loop 正則化
    self_mass = self_attention_mass_from_attn(attn)
    L_self = torch.relu(self_mass - self_target)

    # 全体 loss
    loss = (
        λ_real_eff * L_real_mix         # F2 混合出力の MSE
        + λ_sem      * L_real_sem       # ★ semantic 用の弱い MSE
        + λ_cf_eff   * (L_cf_s + L_cf_p)
        + λ_mono_eff * L_mono
        + λ_cos_eff  * L_cos
        - λ_ent_eff  * H_I
        + λ_env_eff  * L_env
        + λ_self * L_self
        + λ_reverse * L_reverse
    )



    loss.backward()
    opt.step()

    # main の更新後に EMA 更新
    fstat.ema_update(model)

    # F_stat の loss（軽い意味制約）
    L_fstat = fstat_loss(fstat, train_data,
                         base_loss_weight=0.1, meaning_weight=0.05)

    # 200エポックごとに ε & α 更新
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            val_L_sem, _, _ = loss_real_with_attn_F2(model, val_data, alpha=1.0)

        print(
            f"[{epoch}] "
            f"L_real_mix={L_real_mix.item():.4f} "   # F2 混合の MSE
            f"L_real_sem={L_real_sem.item():.4f} "   # semantic 単独の MSE
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


        if prev_state is not None:
            prev_model = MultiIWorldModel(d_model=32, n_heads=4).to(device)
            prev_model.load_state_dict(prev_state)
            prev_model.eval()

            eps_info = epsilon_between_models(
                model_f=model,
                model_g=prev_model,
                data=train_data,
                device=device,
                w_cf=1.0,
                w_mono=1.0,
                w_att=1.0,
                w_self=1.0,
                sym_att=True,
            )

            epsilon_val = eps_info["epsilon"]
            alpha = epsilon_to_alpha(epsilon_val, k=5.0)

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

# -----------------------------
# 6. 反事実チェック & attention 可視化
# -----------------------------
model.eval()
p_tr, s_tr, w_tr, y_tr = train_data

with torch.no_grad():
    # ここでは意味ブランチ（alpha=1）の出力で見る
    _, y_sem, _, _ = model(p_tr, s_tr, w_tr, return_both=True)
    y_real = y_sem
    y_s0   = model(p_tr, torch.zeros_like(s_tr), w_tr)
    y_p0   = model(torch.zeros_like(p_tr), s_tr, w_tr)

print("\n=== 反事実チェック (Multi-I + F2 + self-reg) ===")
print("mean growth (real data) : ", y_real.mean().item())
print("mean growth (sun=0)     : ", y_s0.mean().item())
print("mean growth (plant=0)   : ", y_p0.mean().item())

# attention 平均
with torch.no_grad():
    _, attn, _ = model(p_tr, s_tr, w_tr, return_attn=True)

attn_mean = attn.mean(dim=1).mean(dim=0)  # (7,7)
growth_row = attn_mean[6]                 # (7,)
labels = ["plant", "sun", "water", "I1", "I2", "I3", "growth_self"]

print("\nAvg attention from [growth] query (Multi-I + F2 + self-reg):")
for name, val in zip(labels, growth_row.tolist()):
    print(f"{name:>12}: {val:.4f}")

plt.figure(figsize=(5,4))
plt.bar(labels, growth_row.cpu())
plt.ylim(0,1.0)
plt.title("Avg attention from [growth] query\n(Multi-I + F2 + self-reg)")
plt.tight_layout()
plt.show()
