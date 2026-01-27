# Task 00: 「学習→可視化→崩壊解析」を一本道に統合し、再実行性を高める

## 目的（ゴール）
「可視化と再現性をパッケージ化して、現状の多数スクリプトを『学習→可視化→崩壊解析』の一本道に統合し、再実行性を高める」。

- 1コマンド（または1つのCLI）で、**学習→可視化→崩壊解析**が走り、成果物が規定の場所に揃う
- どの実行でも **同じ入出力規約**・**同じディレクトリ構造**・**同じ設定の保存**がなされ、後から再現できる
- notebook（`src/e_functor.ipynb`）に閉じている解析も、再実行可能なスクリプト/モジュールとして同梱される

---

## 現状コードマップ（全体俯瞰）
※リポジトリ全体は `235 files`。内訳の主成分は `png(94) / csv(63) / py(37) / pt(24)` で、`viz/` と `src/runs/` に生成物が多い。

### ルート直下（ドキュメント）
- `README.md`：研究のコンセプト（α/ε、意味の不変性、崩壊タイプ）概要
- `README_jp.md`：日本語の実行導線（train→可視化→island解析）と解釈メモ
- `SRC_OVERVIEW.md`：`src/` のモジュール説明と簡易フロー
- `SPEC.md`：`src/detect_critical.py`（臨界/崩壊検出）の判定仕様

### `src/`（学習と崩壊検出のコア + 一部解析）
- **コア（学習）**
  - `src/config.py`：ハイパラ・損失重み・デバイス等
  - `src/data.py`：トイ環境データ生成 + split
  - `src/model.py`：`MultiIWorldModel`（7トークン、self-attn、stat/sem 2ヘッド、reverse head）+ `FStatWrapper`(EMA teacher)
  - `src/losses.py`：実データ損失、反事実/単調性、Iのcos抑制、attnエントロピー、self質量、teacher損失
  - `src/metrics.py`：`epsilon_between_models`（dM, dC, d_*）、`epsilon_to_alpha`、健康度指標
  - `src/train.py`：学習本体。`runs/<RUN_ID>/` に `alpha_log.csv` と `islands/island_epoch*.pt` を出力（ただし**CWD依存**で場所が揺れる）
    - 制御は主に環境変数（`SEED`, `RUN_ID`, `RESUME`, `EPS_MODE`, `EPS_OVERRIDE`, `ALPHA_OVERRIDE`, `MUL_*` 等）
- **崩壊/臨界検出（collapse analysis の核）**
  - `src/detect_critical.py`：`alpha_log.csv`→`critical.json/out.json`。baseline・z閾値・Recover/Worsen/Plateau・eps_type 等
  - `src/collect_runs.py`：`runs/*/alpha_log.csv` を巡回して `detect_critical.py` を実行し集計
  - `src/summarize_p_direction.py`：複数runの `p_direction` まとめ（Recover最小条件など）
- **補助解析**
  - `src/island_min_stats.py`：islandスナップショットから要約統計
  - `src/stepLR_cluster_eventrate.py`：`alpha_log.csv`からLR倍率テーブルを提案（`lr_mult_by_epoch.csv`）
  - `src/lr_by_epoch.py`：`lr_mult_by_epoch.csv` を読むユーティリティ
  - `src/attn_common_pca_and_loadings.py`：attnの共通PCA（パスがハードコード気味）
- **テスト**
  - `src/test_policy_gating.py`：`detect_critical` のp_directionロジックの単体テスト
  - `src/test_integration_outjson.py`：CLI仕様が古く、現状の `detect_critical.py` と齟齬がある（要更新）
- **生成物**
  - `src/runs/`：学習結果（`alpha_log.csv`, `model_last.pt`, `islands/*.pt`, `critical.json` 等）
  - `src/__pycache__/`：`.pyc` が含まれている（通常は生成物）

### `old/`（可視化・島解析・各種ステップ解析のスクリプト群）
- 基本可視化：`old/eval_viz.py`, `old/plot_alpha.py`
- island系：`old/island_viz.py`, `old/island_eps.py`, `old/island_profile.py`, `old/island_env_error.py`, `old/island_eps_plot.py`, `old/island_dm_plot.py`, `old/island_dt_by_epoch.py`
- ステップ解析：`old/step_state_series_cluster_B.py`, `old/step2_epsilon_event_embedding.py`, `old/stepC_threshold_update.py`, `old/stepF_sign_event_linear.py`, `old/stepG_cluster_stability.py`, `old/stepG_feature_stability.py`, `old/stepH_semantic_cluster_alignment.py`
- 研究スクリプト断片：`old/attn_pc1_zero_contour_fixed.py`, `old/stable_I_compare.py`
- 旧一体型：`old/experiment.py`（データ生成〜モデル定義〜学習が1ファイルにある旧版）
- 注意：`/content/...` のハードコードが残るものがあり（Colab想定）、**再実行性の阻害要因**

### `src/e_functor.ipynb`（notebookに閉じた解析が存在）
- `viz/stepA_threshold_sources`, `viz/stepB_island_error_explain`, `viz/stepE_*` 等に対応する処理が notebook 側にある（= スクリプトとしては未パッケージ）

### `viz/`（成果物・図・CSV・ptの置き場）
- `viz/` 直下 + `viz/runs/`：図やCSV、`model_last.pt` 等（= 生成物）
- `viz/step*`：各ステップ解析の出力（例：`stepB_state_series_clustering/`, `stepC_threshold_update/`, `stepG_outputs/`, `stepH_semantic_alignment/`）
- 注意：**ソースと成果物が混在**しており、入出力規約が揃っていない

---

## この目的に対する「対象スコープ」
### In-scope（一本道統合に必須）
- 学習：`src/train.py` と依存（`src/config.py`, `src/{data,model,losses,metrics}.py`, `src/lr_by_epoch.py`）
- 可視化：`old/{eval_viz,plot_alpha}.py` + island系（`old/island_*.py`） + notebook内ステップ（`src/e_functor.ipynb` の stepA/B/E 相当）
- 崩壊解析：`src/detect_critical.py` + `src/{collect_runs,summarize_p_direction}.py` + stepC/F/G/H（現状は `old/step*.py` と notebook分散）
- 出力規約：`runs/`・`viz/` の整理、runごとの成果物格納、設定/環境の保存

### Out-of-scope（今回は触らない/後回し）
- 理論の拡張・新規実験設計（研究アイデアの追加）
- 既存の可視化結果（`viz/*.png` 等）の内容そのものの妥当性評価

---

## 現状の再現性を下げている要因（ギャップ）
- **CWD依存の出力**：`src/train.py` の `Path("runs")/...` により、`src/runs/` とルート `runs/` が混在し得る
- **パスのハードコード**：`old/` の一部や notebook が `/content/...` 前提
- **入口が分散**：env var / argparse / notebook が混在し、再実行が「手順依存」になりやすい
- **生成物がリポジトリに混入**：`viz/`・`src/runs/`・`src/__pycache__/` がソースと同列に存在
- **テスト/仕様の齟齬**：`src/test_integration_outjson.py` が `detect_critical.py` の現行CLIと噛み合っていない

---

## タスク一覧（一本道化 + 再現性パッケージ化）
チェックボックスは「実施単位」。上から順にやると一本道に収束しやすい。

### Phase 1: 入口と入出力規約を固定する（最重要）
- [ ] **Runディレクトリ規約を決める**（例：`runs/<run_id>/` を唯一の根拠にする）
  - 成果物は原則 `runs/<run_id>/` 配下に集約（例：`artifacts/`, `figures/`, `collapse/` など）
- [ ] **共通のパス解決を実装**（「常にrepoルート基準」or「常に`--run-dir`基準」を統一）
- [ ] **設定のスナップショット保存**（実行時設定を `runs/<run_id>/config.json` などに保存）
  - `SEED`/loss重み/ハイパラ/介入（`EPS_MODE`等）/依存バージョン情報の記録
- [ ] **再現性メタデータ保存**（Python/torch/numpy/pandas/sklearn の version、実行コマンド、環境変数）

### Phase 2: 学習を「再実行可能なコマンド」にする
- [ ] `src/train.py` を **env var依存からCLI引数 + config** に寄せる（互換のため env var は残してもOK）
- [ ] 学習出力の場所を **`--run-dir`（または `--run-id`）で明示**し、CWD依存を排除
- [ ] `runs/<run_id>/` に必ず `alpha_log.csv`, `model_last.pt`, `islands/*.pt` が揃うことを保証
- [ ] 既存の `RESUME`/重複epoch回避/ログスキーマ（`LOG_FIELDS`）の仕様を文書化し固定

### Phase 3: 可視化を「パッケージ化」する（学習→可視化の一本化）
- [ ] `old/eval_viz.py` / `old/plot_alpha.py` を `run_dir` 入力で動くように統一（出力先も `run_dir` 配下へ）
- [ ] island可視化（`old/island_*`）を `run_dir` 入力で動くように統一
- [ ] notebook 依存の stepA/stepB_island_error_explain/stepE を **.pyへ抽出**し、`run_dir` から生成可能にする
- [ ] 主要図の「最小セット」を定義（例：`alpha/epsilon`推移、寄与率、attn bar、island時系列、threshold mode）

### Phase 4: 崩壊解析（collapse）を一本道の後半に接続する
- [ ] `src/detect_critical.py` を **library関数 + CLI** に分離（呼び出し側がsubprocessに頼らない設計へ）
- [ ] `critical.json/out.json` の出力場所を `runs/<run_id>/collapse/` に統一（命名も固定）
- [ ] `src/collect_runs.py` / `src/summarize_p_direction.py` を「複数run比較」用のサブコマンドとして整理
- [ ] stepC/F/G/H（`old/step*.py`）のI/Oを `runs/<run_id>/derived/...` に統一し、必要入力を明文化

### Phase 5: 「一本道パイプライン」実行器を作る
- [ ] `pipeline` コマンド（例：`python -m efunctor pipeline --run-id ...`）を作り、
  - `train → viz_basic → island_* → detect_critical → step*` を順次実行
  - `--stages train,viz,collapse` のように部分実行も可能にする
  - 既存成果物がある場合はスキップ/再計算（`--recompute`）を制御できるようにする

### Phase 6: 依存関係・実行方法を固定（再現性の最後のピース）
- [ ] `pyproject.toml` か `requirements.txt` を整備し、最低限の依存（torch/numpy/pandas/matplotlib/sklearn/pytest）を明記
- [ ] `README_jp.md` に「一本道の実行手順（1〜3コマンド）」を追記し、古い導線（`old/` 直叩き・notebook依存）を整理

### Phase 7: 生成物の扱いを整理（再現性とメンテ性のため）
- [ ] `viz/` と `src/runs/` の位置づけを決める（例：`artifacts/` に移す / `.gitignore` する / `examples/` に縮退）
- [ ] `src/__pycache__/` や `.pyc` をリポジトリから除外し、`.gitignore` を追加

### Phase 8: テストで再現性を担保する
- [ ] `src/test_integration_outjson.py` を現行CLI（`detect_critical.py <csv> --json_out <path>`）に合わせて修正
- [ ] 「学習→ログ生成→検出」の最小スモークテスト（短いepoch/小N）を追加して、CI相当で崩れないようにする

---

## 完了条件（Definition of Done）
- `runs/<run_id>/` に **学習ログ・モデル・islandスナップショット・主要可視化・崩壊解析JSON** が揃う
- ルート直下/`src/`/`old/` を跨いだ手作業なしで、`run_id` を指定して再実行できる
- notebookだけに存在する重要解析（stepA/B/E）が、スクリプトとして再実行できる

