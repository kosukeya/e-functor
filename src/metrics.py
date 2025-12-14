# metrics.py
import torch
import torch.nn.functional as F
from losses import entropy_I_attention

@torch.no_grad()
def pearson_corr(a, b, eps=1e-8):
    a = a.view(-1)
    b = b.view(-1)
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (a0.std(unbiased=False) * b0.std(unbiased=False) + eps)
    return (a0 * b0).mean() / denom

@torch.no_grad()
def attention_distribution_with_self(model, data):
    p, s, w, _ = data
    _, attn, _ = model(p, s, w, return_attn=True)
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
    return torch.relu(y1 - y2).mean().item()

@torch.no_grad()
def dM(model_f, model_g, data, device,
       w_cf=1.0, w_mono=1.0, w_att=1.0, w_self=1.0,
       sym_att=True):

    CF_f = cf_violation(model_f, data)
    CF_g = cf_violation(model_g, data)
    d_cf = abs(CF_f - CF_g)

    MONO_f = mono_violation(model_f, device=device, n_pairs=128)
    MONO_g = mono_violation(model_g, device=device, n_pairs=128)
    d_mono = abs(MONO_f - MONO_g)

    dist_f = attention_distribution_with_self(model_f, data)
    dist_g = attention_distribution_with_self(model_g, data)

    p = dist_f[:6]; q = dist_g[:6]
    p = p / p.sum(); q = q / q.sum()

    if sym_att:
        d_att = KL_divergence(p, q) + KL_divergence(q, p)
    else:
        d_att = KL_divergence(p, q)

    self_f = float(dist_f[6].item())
    self_g = float(dist_g[6].item())
    d_self = abs(self_f - self_g)

    dM_val = w_cf * d_cf + w_mono * d_mono + w_att * d_att + w_self * d_self
    return {"d_cf": d_cf, "d_mono": d_mono, "d_att": d_att, "d_self": d_self,
            "self_f": self_f, "self_g": self_g, "dM": dM_val}

@torch.no_grad()
def epsilon_between_models(model_f, model_g, data, device,
                           w_cf=1.0, w_mono=1.0, w_att=1.0, w_self=1.0,
                           sym_att=True):
    dC_val = dC(model_f, model_g, data)
    dM_dict = dM(model_f, model_g, data, device,
                 w_cf=w_cf, w_mono=w_mono, w_att=w_att, w_self=w_self,
                 sym_att=sym_att)
    eps = dM_dict["dM"] - dC_val
    return {"dC": dC_val, **dM_dict, "epsilon": eps}

def epsilon_to_alpha(epsilon, k=5.0):
    import math
    eps_clipped = max(min(epsilon, 1.0), -1.0)
    return 1.0 / (1.0 + math.exp(-k * eps_clipped))

@torch.no_grad()
def semantic_health_metrics(model, data, device):
    p, s, w, y = data

    _, y_sem, attn, _ = model(p, s, w, return_both=True)
    y_sem = y_sem.detach()
    y_mean = float(y_sem.mean().item())
    y_std  = float(y_sem.std(unbiased=False).item())
    y_min  = float(y_sem.min().item())
    y_max  = float(y_sem.max().item())
    corr   = float(pearson_corr(y_sem, y).item())

    s0 = torch.zeros_like(s)
    p0 = torch.zeros_like(p)
    _, y_sem_s0, _, _ = model(p, s0, w, return_both=True)
    _, y_sem_p0, _, _ = model(p0, s, w, return_both=True)

    m_real = float(y_sem.mean().item())
    m_s0   = float(y_sem_s0.mean().item())
    m_p0   = float(y_sem_p0.mean().item())

    gap_s = m_real - m_s0
    gap_p = m_real - m_p0
    ratio_s = m_s0 / (m_real + 1e-8)
    ratio_p = m_p0 / (m_real + 1e-8)

    attn_mean = attn.mean(dim=1).mean(dim=0)
    g_row = attn_mean[6]
    env_sum  = float(g_row[0:3].sum().item())
    I_sum    = float(g_row[3:6].sum().item())
    self_m   = float(g_row[6].item())
    H_I_val  = float(entropy_I_attention(attn).item())

    return dict(
        y_mean=y_mean, y_std=y_std, y_min=y_min, y_max=y_max,
        corr=corr,
        m_real=m_real, m_s0=m_s0, m_p0=m_p0,
        gap_s=gap_s, gap_p=gap_p,
        ratio_s=ratio_s, ratio_p=ratio_p,
        env_sum=env_sum, I_sum=I_sum, self_m=self_m,
        H_I=H_I_val,
    )
