# config.py
import torch

# ---- Collapse guards (minimal) ----
LAMBDA_SEM_MIN = 0.20   # semantic回帰の下限（αに依らず確保）
CF_HINGE_MARGIN = 1e-3  # 反事実 hinge のマージン
LOGSPACE_EPS = 1e-8     # log-space用

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Training hyperparams ----
N = 5000
TRAIN_RATIO = 0.8
EPOCHS = 2001
LR = 1e-3

D_MODEL = 32
N_HEADS = 4

# loss weights
LAMBDA = dict(
    cf_base=1.0,
    mono_base=0.5,
    cos_base=0.1,
    ent_base=0.05,
    env_base=0.5,
    real_base=1.0,
    self_=1.0,
    sem=0.1,
    reverse=0.01,
)

SELF_TARGET = 0.03

# EMA teacher
EMA_TAU = 0.02
FSTAT_SCALE = 2.0

# reverse head
REV_MARGIN = 1.0

# epsilon->alpha
ALPHA_K = 5.0

# logging
LOG_EVERY = 200
