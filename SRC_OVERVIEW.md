# `src/` ディレクトリ概要

本ドキュメントは、`src/` 配下にある各モジュールの目的、処理内容、および主な実行結果の想定を README から参照できるようにまとめたものです。

## 共通設定: `config.py`
- 学習用ハイパーパラメータ（エポック数、学習率、モデル次元など）と各種ロスの重みを定義。
- デバイス選択（GPU/CPU）やログ出力頻度、EMA teacher 用の係数などを集約し、他モジュールが参照して一貫した実験設定を行う。

## データ生成と分割: `data.py`
- `generate_world_data(n)`: 植物の有無・日照量・水量の乱数から成長量を計算する簡易シミュレーションデータを生成。
- `make_split(n, train_ratio, seed)`: データインデックスを学習/検証に分割。
- `pack_data(...)`: 指定インデックスのテンソルをまとめてデバイスへ転送し、学習ループに投入できる形に整える。

## モデル定義: `model.py`
- `MultiHeadSelfAttention`: トークン系列に対するシンプルな自前実装の多頭自己注意層。
- `MultiIWorldModel`: `[plant, sun, water, I1, I2, I3, growth]` の7トークンを入力とし、
  - 環境埋め込みから3つの中間トークン `I1-3` を生成しつつ自己注意で情報を混合。
  - `head_sem` (因果的/semantic) と `head_stat` (統計的/statistical) の2出力ヘッドで成長量を推定し、`return_both=True` で両方を返す。
  - `reverse_from_growth` は growth トークンから環境要因を推定するリバースヘッドを提供。
- `FStatWrapper`: メインモデルの EMA 版 teacher。スケール制御を掛けた出力を返しつつ、`ema_update` で重みを更新する。

## 損失関数群: `losses.py`
- 実データに対する MSE（stat/sem 混合）、対数空間でのセマンティック回帰、反事実・単調性制約、自己注意のエントロピー/環境寄与ペナルティなどを実装。
- 埋め込みトークン間のコサイン類似度抑制 `cosine_divergence_I` や、teacher (FStat) 学習用ロス `fstat_loss` も提供し、学習ループが複合目的を組み合わせられるようにしている。

## 評価指標: `metrics.py`
- ピアソン相関、反事実/単調性違反、注意分布の KL ダイバージェンスに基づく `dM`/`epsilon` 計算など、モデル安定性を測るユーティリティ。
- `semantic_health_metrics` ではセマンティック出力の統計量・反事実ギャップ・注意配分をまとめて返し、トレーニングログで健康度を確認できる。

## 学習スクリプト: `train.py`
- 擬似データ生成 → モデル/最適化器/EMA teacher 初期化後、複数ロスを重み付きで合成して学習。
- エポックごとに `epsilon_between_models` を用いて直前エポックとのモデル差分からゲート係数 `alpha` を自動更新し、semantic/statistical の比率を動的制御。
- 定期的にメトリクスを標準出力と `runs/alpha_log.csv` に記録し、`runs/islands/` に I 埋め込みと attention 行を含むスナップショットを保存。終了時に重みを `model_last.pt` として出力。
- 想定される実行結果: ログには MSE、制約ロス、相関係数、`epsilon`、`alpha` 推移などが出力され、`runs/` 下に CSV とチェックポイント（`model_last.pt`、island スナップショット）が生成される。

## 可視化・解析スクリプト
- `eval_viz.py`: 学習済み重み `model_last.pt` を読み込み、semantic/statistical/混合出力の平均や相関、反事実シナリオ（sun=0, plant=0）の平均予測を標準出力に表示。寄与度バーと注意バーの PNG（`branch_contrib.png`、`attn_bar.png`）を保存。
- `plot_alpha.py`: `runs/alpha_log.csv` を柔軟に読み込み、αやε、`dM`/`dC` 推移、`dM` 内訳を折れ線で描画し、`runs/` 配下に PNG を保存（欠損列はスキップ）。
- `island_viz.py`: island スナップショット (`runs/islands/island_epoch*.pt`) をクラスタリングし、エポック間でラベルを整合させつつ PCA 散布図、滞留時間統計（`island_dwell.csv` など）、島ごとの反事実ギャップ平均を出力。クラスタ数はシルエットスコアで自動選択可。
- `island_eps.py`: island スナップショット群を読み込み、エポック間でクラスタラベルを揃えた上で各島ごとに `epsilon_between_models` を計算し、総合および島別の ε/dM/dC 内訳を `island_eps_summary.csv` として出力。

## 簡易実行フローまとめ
1. `python src/train.py` でモデルを学習し、`runs/` と `model_last.pt` を生成。
2. `python src/eval_viz.py` で学習済みモデルの分岐性能・注意分布を可視化（PNG 出力）。
3. `python src/plot_alpha.py` で学習ログから α/ε などの推移グラフを生成。
4. `python src/island_viz.py --use_attn` で island スナップショットをクラスタリングし、滞留時間やクラスタ別反事実差分を可視化。
5. `python src/island_eps.py` でエポック間の島ごとの安定性 (ε, dM, dC) を計算し CSV にまとめる。
