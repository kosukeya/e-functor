# data.py
import torch

def generate_world_data(n: int):
    plant = torch.bernoulli(torch.full((n,), 0.7))
    sun   = torch.rand(n)
    water = torch.rand(n)

    base = 1.2 * sun + 0.8 * water
    base = torch.tanh(base * 2.0)
    noise = 0.05 * torch.randn(n)
    growth = plant * (base + noise)
    growth = torch.clamp(growth, min=0.0)
    return plant, sun, water, growth

def make_split(n: int, train_ratio: float = 0.8, seed: int | None = None):
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
    else:
        perm = torch.randperm(n)

    n_tr = int(train_ratio * n)
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:]
    return train_idx, val_idx

def pack_data(plant, sun, water, growth, idx, device: str):
    return (
        plant[idx].to(device),
        sun[idx].to(device),
        water[idx].to(device),
        growth[idx].to(device),
    )
