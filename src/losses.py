# losses.py
import torch
import torch.nn.functional as F
from config import CF_HINGE_MARGIN, LOGSPACE_EPS

def loss_real_with_attn_F2(model, data, alpha: float):
    p, s, w, y = data
    y_stat, y_sem, attn, I_triplet = model(p, s, w, return_both=True)
    y_pred = (1.0 - alpha) * y_stat + alpha * y_sem
    mse = F.mse_loss(y_pred, y)
    return mse, attn, I_triplet

def cf_hinge_loss(y, margin=1e-3):
    return (torch.relu(torch.abs(y) - margin) ** 2).mean()

def loss_real_sem_log(model, data, eps=LOGSPACE_EPS):
    p, s, w, y = data
    _, y_sem, _, _ = model(p, s, w, return_both=True)
    return F.mse_loss(torch.log(y_sem + eps), torch.log(y + eps))

def loss_counterfactual_sun0_sem(model, data):
    p, s, w, _ = data
    s0 = torch.zeros_like(s)
    _, y_sem, _, _ = model(p, s0, w, return_both=True)
    return cf_hinge_loss(y_sem, margin=CF_HINGE_MARGIN)

def loss_counterfactual_plant0_sem(model, data):
    p, s, w, _ = data
    p0 = torch.zeros_like(p)
    _, y_sem, _, _ = model(p0, s, w, return_both=True)
    return cf_hinge_loss(y_sem, margin=CF_HINGE_MARGIN)

def loss_monotonic_sun_sem(model, device, n_pairs=64):
    plant = torch.ones(n_pairs, device=device)
    water = torch.rand(n_pairs, device=device)
    sun1  = torch.rand(n_pairs, device=device)
    sun2  = sun1 + torch.rand(n_pairs, device=device) * (1.0 - sun1)
    _, y_sem1, _, _ = model(plant, sun1, water, return_both=True)
    _, y_sem2, _, _ = model(plant, sun2, water, return_both=True)
    return torch.relu(y_sem1 - y_sem2).mean()

def cosine_divergence_I(I_triplet):
    I1, I2, I3 = I_triplet
    def cos(a, b):
        a_flat = a.view(a.size(0), -1)
        b_flat = b.view(b.size(0), -1)
        return F.cosine_similarity(a_flat, b_flat, dim=-1)
    c12 = cos(I1, I2)
    c13 = cos(I1, I3)
    c23 = cos(I2, I3)
    return (c12**2 + c13**2 + c23**2).mean()

def entropy_I_attention(attn):
    attn_mean = attn.mean(dim=1)          # (B,7,7)
    growth_row = attn_mean[:, 6, :]       # (B,7)
    probs_I = growth_row[:, 3:6]
    eps = 1e-8
    log_p = torch.log(probs_I + eps)
    ent = -(probs_I * log_p).sum(dim=-1)
    return ent.mean()

def env_attention_penalty(attn, alpha=0.3):
    attn_mean = attn.mean(dim=1)
    growth_row = attn_mean[:, 6, :]
    env_sum = growth_row[:, 0:3].sum(dim=-1)
    penalty = torch.relu(alpha - env_sum)
    return penalty.mean()

def self_attention_mass_from_attn(attn):
    attn_mean = attn.mean(dim=1)
    growth_row = attn_mean[:, 6, :]
    self_mass = growth_row[:, 6]
    return self_mass.mean()

def fstat_loss(fstat, data, device, base_loss_weight=0.1, meaning_weight=0.05):
    p, s, w, _ = data
    y_pred = fstat(plant=p, sun=s, water=w)
    L_basic = (y_pred ** 2).mean()

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

    return base_loss_weight * L_basic + meaning_weight * (L_cf + L_mono)
