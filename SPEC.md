# SPEC.md — Criticality Detector 判定仕様（v0.9 / 短い運用版）

この文書は `detect_critical.py` が `alpha_log.csv` を解析して生成する `out.json` の
判定ポリシー（Normal/Abnormal、Recover/Worsen/Plateau、eps_type 等）を短く明文化する。

---

## 0. 目的

学習ログ `alpha_log.csv` から次を機械判定する：

- **どの種類の異常が起きたか**（env/self/eps）
- **異常後に改善したか**（Recover）／**悪化したか**（Worsen）／**停滞か**（Plateau）
- **異常の主因が何か**（eps_type / top2）

目的は「人間が読みやすい要約」と「run 間比較の安定化」である。

---

## 1. 入力

### 1.1 必須ログ列（alpha_log.csv）
最低限、以下の列が必要：

- `epoch`
- `epsilon`
- `dC`
- `d_cf`
- `d_mono`
- `d_att`
- `d_self`

---

## 2. ベースライン（正常域の定義）

### 2.1 baseline window
原則：

- baseline は epochs **[200, 600]** を対象とする

ただし対象行数が少なすぎる場合：

- baseline rows が不足 → **first 5 rows** にフォールバックする

（例：`[WARN] Baseline window too small ... Falling back to first 5 rows baseline.`）

### 2.2 baseline 統計量
baseline から以下を推定する：

- `z_meds`（中央値）
- `z_sigs`（sigma 相当のスケール）

これらは `out.json` の以下に出力される：

- `diag.self.z_meds`
- `diag.self.z_sigs`

---

## 3. 異常検出（Critical Events）

### 3.1 閾値
- `z_soft = 3.0`
- `z_hard = 4.0`

### 3.2 イベント種類（events.kind）
検出対象：

- `env_break`
- `self_break`
- `self_shock`

### 3.3 異常とみなす条件（単発ノイズ抑制）
- `z > z_soft` → 異常候補
- `z > z_hard` → 強い異常候補

単発ノイズによる誤検出を避けるため、以下を併用する：

- hard は **2 consecutive**
- soft は **2 of 3**

（`out.json` の `2consec_hard` / `2of3_soft` に対応）

---

## 4. Recover/Worsen/Plateau 判定ポリシー（重要）

### 4.1 大原則：Recover は「異常域のみ」許可
**異常が起きていない run（Normal）で Recover を出してはいけない。**

Normal run の期待挙動：

- `events = []`
- `p_peak_abnormal = false`
- `p_direction = Plateau`
- `eps_type = NORMAL`
- `p_eps_series_P_valid` は全て 0.0（または None/0）

---

## 5. Self側（p_direction）の定義

Self 系の系列 `P(t)` は、baseline を除いた valid epoch の primary z 系列から作る。
（例：`p_series_z_primary_valid`）

以下を算出する：

- `p_peak_epoch`, `p_peak`：最大値点
- `p_tail_epoch`, `p_tail`：末尾代表点（最後の有効点）
- `p_tail_minus_peak`
- `p_slope_after_peak`

### 5.1 判定ルール（Self）
- `p_peak_abnormal == False` の場合  
  → **p_direction = Plateau（強制）**

- `p_peak_abnormal == True` の場合のみ  
  → Recover/Worsen/Plateau の通常判定を許可する

---

## 6. ε成分側（p_eps_direction）の定義

ε成分側の系列 `P_eps(t)` は、各 epoch の z から **正の超過分** を作る：

- `P_eps = max(0, z - z_soft)`

この定義により：

- z が負 → P_eps = 0
- z が微小 → P_eps = 0

となり、Normal なのに Recover が出る事故を防ぐ。

### 6.1 判定ルール（eps）
- `p_eps_peak_abnormal == False`  
  → `p_eps_direction = Plateau（強制）`

- `p_eps_peak_abnormal == True`  
  → Recover/Worsen/Plateau の通常判定を許可する

---

## 7. eps_type（異常の主因ラベル）

### 7.1 出力方針
- 異常がないとき：`eps_type = NORMAL`
- 異常があるときのみ：`eps_type = {RULE/FRAME/SELF/...}`

### 7.2 top2 表示
異常時のみ：

- `eps_type_top2 = "TYPE1:score | TYPE2:score"`

を出力する（原因候補を2つ提示）。

---

## 8. hidden / behavior 判定（現状の位置づけ）

- `eps_hidden`
- `eps_behavior`

は追加のメタ分類（安全装置）であり、例えば：

- 異常があるが「表に出すべきでない」
- 異常が「行動的に危険」

などの判定のためのフラグである。

現時点では `false` が続く想定でよく、今後テストで詰める。

---

## 9. 運用上の期待値（成功条件）

### 9.1 no-intervention run（正常系）
- `events = []`
- `p_peak_abnormal = false`
- `p_direction = Plateau`
- `eps_type = NORMAL`
- `p_eps_series_P_valid` が全て 0.0（または None/0）

### 9.2 C1 など介入あり run（異常系）
- `env_break` が出る（例：epoch 400）
- `self_break` / `self_shock` が出ることがある（後半）
- `p_peak_abnormal = true`
- `p_direction = Recover`（異常域でのみ許可）
- `eps_type = SELF` などが出る（異常時のみ）

---

## 10. 次のステップ（テスト追加）
この仕様に対して境界ケーステストを追加する：

1. Normal run で Recover が絶対に出ない  
2. 異常 run でだけ Recover/Worsen が出る  
3. p_eps は負や微小で Recover が出ない（0 に潰れる）
