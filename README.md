# プロジェクト概要
本リポジトリは、簡易な「植物の成長」シミュレーションデータ上で、統計的な回帰と意味的（因果的）な推論を切り替える "ε-関手" 的なゲート制御を実験するコードです。MultiIWorldModel の semantic/statistical 2系統の出力を、エポック間の距離 `ε` から算出したゲート係数 `α` で動的に混合し、反事実・単調性制約を課しながら安定性を評価します。

## コード構成のハイライト
- **データ生成（`src/data.py`）**: 植物フラグ・日照・水量から成長量を計算するトイデータを作成し、学習/検証に分割します。
- **モデル（`src/model.py`）**: 7 トークン `[plant, sun, water, I1, I2, I3, growth]` に自己注意を適用し、semantic/statistical の2ヘッドで成長を予測。EMA 版の FStatWrapper も併設しています。
- **損失・メトリクス（`src/losses.py`, `src/metrics.py`）**: F2 混合損失、反事実・単調性制約、自己注意のエントロピー/環境寄与ペナルティ、`epsilon_between_models` などの距離指標を実装。
- **学習スクリプト（`src/train.py`）**: 2001 エポックの学習で `α` を自動更新しつつログを `runs/alpha_log.csv` に保存。各ステップの semantic 健康度や寄与率、反事実ギャップに加え、島スナップショット（`runs/islands/island_epoch*.pt`）を生成します。
- **可視化（`src/eval_viz.py`, `src/plot_alpha.py`, `src/island_viz.py`, `src/island_eps.py`）**: 学習済み重みを読み込み、ブランチ寄与とアテンション、`α/ε` 推移、島クラスタの遷移や距離分解をプロットします。

より詳細なファイル別の説明は `SRC_OVERVIEW.md` を参照してください。

## 実行方法
1. 学習: `python src/train.py`
   - `runs/alpha_log.csv` にメトリクスが追記され、最終重みが `model_last.pt` として保存されます。
   - `runs/islands/` 配下に I 埋め込み・注意行・反事実予測を含むスナップショットがエポックごとに保存されます。
2. 評価/可視化: `python src/eval_viz.py`
   - semantic/statistical/混合出力の平均・相関、反事実シナリオの平均、寄与率バー（`branch_contrib.png`）、成長トークンからの注意バー（`attn_bar.png`）を出力します。
3. ログの推移確認: `python src/plot_alpha.py`
   - `runs/alpha_log.csv` から α/ε や dM/dC の折れ線グラフを生成し、PNG を保存します（列が存在する場合のみ描画）。
4. 島のクラスタ解析: `python src/island_viz.py --use_attn`
   - `runs/islands/island_epoch*.pt` をクラスタリングし、PCA 散布図や滞留時間、島ごとの反事実ギャップを出力します。
5. 島間距離の分解: `python src/island_eps.py`
   - エポック間の island スナップショットから ε, dM, dC を集計し、CSV/PNG にまとめます。

## 参考出力（`viz/` フォルダ）
リポジトリには直近の実行結果として以下の図が含まれています。`src/eval_viz.py` や island 系スクリプトで再生成できます。
- `branch_contrib.png`, `attn_bar.png`: ブランチ寄与率と成長トークンからの平均注意。
- `Island decision space (epoch=2000) 1.png`, `Island decision space (epoch=2000) 2.png`: island クラスタの決定境界例。
- `island_dt_by_epoch_err_abs_mean.csv`, `island_dt_threshold_timeseries.png`, `err_abs_mean_by_island_timeseries.png`: 島クラスタの遷移と誤差推移を表す表と時系列プロット。

## リポジトリの目的
統計的なフィットと意味的（因果的）な制約を併用し、エポック間距離 `ε` を通じて2系統の寄与を自動調整する仕組みを検証することを目的としています。トイ環境での挙動を可視化・分解するための各種ロガーと解析ツールが揃っています。

## 境界回転と訓練制御量の関係（確定記述）

本解析では、attention 主成分 attn_PC1=0 により定義される境界が、I-PCA 空間上で固定された幾何学的超平面ではなく、訓練過程に応じて回転する可動境界であることを示した。

特に、境界の回転角 θ は epsilon の変化量と強い一次関係を示し、差分相関において Δθ と −Δε の間にほぼ完全な線形相関（r≈−0.995）が観測された。一方、同じ回転量と alpha_used の変化量との間には有意な対応は見られなかった。

この結果は、境界の回転が主として exploration 制御量 epsilon によって駆動され、alpha_used は境界の回転そのものではなく、回転後の attention 配分や位置調整に関与する補助的役割を担っていることを示唆する。

したがって、attn_PC1=0 境界は「表現空間内に固定された判別境界」ではなく、「訓練制御量に応じて回転する注意制御的境界」として理解されるべきである。
