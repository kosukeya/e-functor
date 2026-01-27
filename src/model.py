# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MultiIWorldModel(nn.Module):
    """
    トークン列: [plant, sun, water, I1, I2, I3, growth]
    """
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        self.d_model = d_model

        self.embed_in = nn.Linear(1, d_model)

        self.I_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 3 * d_model),
        )

        self.growth_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)

        self.head_sem = nn.Linear(d_model, 1)
        self.head_stat = nn.Linear(d_model, 1)
        self.head_reverse = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 3),
        )

    def forward(self, plant, sun, water, return_attn=False, return_both=False):
        B = plant.shape[0]

        p = self.embed_in(plant.view(B, 1, 1))
        s = self.embed_in(sun.view(B, 1, 1))
        w = self.embed_in(water.view(B, 1, 1))

        env = torch.stack([plant, sun, water], dim=-1)  # (B,3)
        I_all = self.I_mlp(env).view(B, 3, self.d_model)
        I1, I2, I3 = I_all[:, 0:1, :], I_all[:, 1:2, :], I_all[:, 2:3, :]

        g = self.growth_token.expand(B, -1, -1)
        x = torch.cat([p, s, w, I1, I2, I3, g], dim=1)  # (B,7,D)

        # mask
        T = x.size(1)
        base = torch.zeros((T, T), device=x.device, dtype=x.dtype)
        upper = torch.triu(torch.ones((T, T), device=x.device), diagonal=1).bool()
        base = base.masked_fill(upper, float("-inf"))
        base[3:6, :] = float("-inf")
        base[3:6, 0:3] = 0.0
        attn_mask = base.unsqueeze(0).unsqueeze(0).expand(B, self.attn.n_heads, T, T).clone()

        x_out, attn = self.attn(x, attn_mask=attn_mask, need_weights=True)

        # semantic
        growth_sem_rep = x_out[:, 6, :]
        growth_sem_raw = self.head_sem(growth_sem_rep).squeeze(-1)
        growth_sem_pred = F.softplus(growth_sem_raw) + 1e-6

        # statistical
        env_reps = x_out[:, 0:3, :]
        env_mean = env_reps.mean(dim=1)
        growth_stat_raw = self.head_stat(env_mean).squeeze(-1)
        growth_stat_pred = F.softplus(growth_stat_raw)

        # after growth_sem_pred and growth_stat_pred
        gate = (plant > 0).to(growth_sem_pred.dtype)  # (B,)
        growth_sem_pred = growth_sem_pred * gate
        growth_stat_pred = growth_stat_pred * gate

        if return_both:
            return growth_stat_pred, growth_sem_pred, attn, (I1, I2, I3)
        if return_attn:
            return growth_sem_pred, attn, (I1, I2, I3)
        return growth_sem_pred

    def reverse_from_growth(self, batch_size, device=None):
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
    出力にスケール制御を入れる。
    """
    def __init__(self, model: MultiIWorldModel, device: str, tau=0.02, scale=2.0):
        super().__init__()
        self.teacher = MultiIWorldModel(d_model=model.d_model, n_heads=model.attn.n_heads).to(device)
        self.teacher.load_state_dict(model.state_dict())
        self.tau = tau
        self.scale = scale

    @torch.no_grad()
    def ema_update(self, model: MultiIWorldModel):
        for p_teacher, p_main in zip(self.teacher.parameters(), model.parameters()):
            p_teacher.data = (1 - self.tau) * p_teacher.data + self.tau * p_main.data

    def forward(self, plant, sun, water):
        raw = self.teacher(plant, sun, water)
        y = torch.tanh(raw) * self.scale
        y = F.softplus(y)
        return y
