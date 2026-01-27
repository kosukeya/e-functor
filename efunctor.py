# efunctor.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_utils import resolve_run_dir, build_run_paths, ensure_run_layout


def _strip_args(argv: List[str], keys: Iterable[str]) -> List[str]:
    keys = set(keys)
    out: List[str] = []
    skip_next = False
    for i, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if tok in keys:
            skip_next = True
            continue
        out.append(tok)
    return out


def _should_run(recompute: bool, outputs: Iterable[Path]) -> bool:
    if recompute:
        return True
    outputs = list(outputs)
    if not outputs:
        return True
    return any(not p.exists() for p in outputs)


def run_pipeline(argv: List[str]) -> None:
    # Force non-interactive backend for pipeline runs (avoid Tk errors on exit).
    if "MPLBACKEND" not in os.environ:
        os.environ["MPLBACKEND"] = "Agg"

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--stages", type=str, default="train,viz,island,collapse,steps")
    ap.add_argument("--recompute", action="store_true")
    args, train_rest = ap.parse_known_args(argv)

    stages_raw = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages_raw or "all" in stages_raw:
        stages = {"train", "viz", "island", "collapse", "steps"}
    else:
        stages = set()
        for s in stages_raw:
            if s == "viz":
                stages.add("viz")
            elif s == "viz_basic":
                stages.add("viz")
            elif s in {"feature_stability", "stepG_feature_stability", "stepG"}:
                stages.add("feature_stability")
            else:
                stages.add(s)

    run_dir, run_id = resolve_run_dir(args.run_dir, args.run_id)
    paths = build_run_paths(run_dir, run_id)
    ensure_run_layout(paths)

    train_rest = _strip_args(train_rest, {"--run-id", "--run-dir", "--config"})

    if "train" in stages:
        from train import main as train_main

        outputs = [paths.alpha_log, paths.model_last]
        if _should_run(args.recompute, outputs):
            train_args = ["--run-dir", str(paths.run_dir), "--run-id", run_id]
            if args.config:
                train_args += ["--config", args.config]
            train_args += train_rest
            print("[pipeline] train")
            train_main(train_args)
        else:
            print("[pipeline] skip train (outputs exist)")

    if "viz" in stages:
        from old.plot_alpha import main as plot_alpha_main
        from old.eval_viz import main as eval_viz_main

        out_dir = paths.figures_dir / "viz_basic"
        outputs = [out_dir / "alpha.png", out_dir / "attn_bar.png", out_dir / "branch_contrib.png"]
        if _should_run(args.recompute, outputs):
            print("[pipeline] viz_basic")
            plot_alpha_main(["--run-dir", str(paths.run_dir)])
            eval_viz_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip viz_basic (outputs exist)")

    if "island" in stages:
        from old.island_viz import main as island_viz_main
        from old.island_env_error import main as island_env_error_main
        from old.island_profile import main as island_profile_main
        from old.island_eps import main as island_eps_main
        from old.island_eps_plot import main as island_eps_plot_main
        from old.island_dm_plot import main as island_dm_plot_main
        from old.island_dt_by_epoch import main as island_dt_main

        if _should_run(args.recompute, [paths.derived_dir / "island_viz" / "island_summary.csv"]):
            print("[pipeline] island_viz")
            island_viz_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip island_viz (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "island_env_error.csv"]):
            print("[pipeline] island_env_error")
            island_env_error_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip island_env_error (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "island_profile.csv"]):
            print("[pipeline] island_profile")
            island_profile_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip island_profile (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "island_eps_summary.csv"]):
            print("[pipeline] island_eps")
            island_eps_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip island_eps (outputs exist)")

        eps_fig_dir = paths.figures_dir / "island_eps"
        if _should_run(args.recompute, [eps_fig_dir / "island_eps_timeseries.png"]):
            print("[pipeline] island_eps_plot")
            island_eps_plot_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip island_eps_plot (outputs exist)")

        if _should_run(args.recompute, [eps_fig_dir / "island_dM_components_timeseries.png"]):
            print("[pipeline] island_dm_plot")
            island_dm_plot_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip island_dm_plot (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "island_dt_by_epoch_err_abs_mean.csv"]):
            print("[pipeline] island_dt_by_epoch")
            island_dt_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip island_dt_by_epoch (outputs exist)")

    if "collapse" in stages:
        from detect_critical import main as detect_main

        outputs = [paths.collapse_dir / "critical.json"]
        if _should_run(args.recompute, outputs):
            print("[pipeline] detect_critical")
            detect_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip detect_critical (outputs exist)")

    if "steps" in stages:
        import stepA_threshold_sources as stepA
        import stepB_island_error_explain as stepB
        import stepD_threshold_update_events as stepD
        import stepE_threshold_modes as stepE_modes
        import stepE_event_prepost as stepE_prepost
        from old.step_state_series_cluster_B import main as step_state_main
        from old.step2_epsilon_event_embedding import main as step2_main
        from old.stepC_threshold_update import main as stepC_main
        from old.stepF_sign_event_linear import main as stepF_main

        if _should_run(args.recompute, [paths.derived_dir / "threshold_timeseries_with_events.csv"]):
            print("[pipeline] stepD_threshold_update_events")
            stepD.main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip stepD_threshold_update_events (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "stepA_threshold_sources" / "threshold_candidate_correlations.csv"]):
            print("[pipeline] stepA_threshold_sources")
            stepA.main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip stepA_threshold_sources (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "stepB_island_error_explain" / "stepB_model_summary.csv"]):
            print("[pipeline] stepB_island_error_explain")
            stepB.main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip stepB_island_error_explain (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "stepB_state_series_clustering" / "trajectory_cluster_assignments.csv"]):
            print("[pipeline] step_state_series_cluster_B")
            step_state_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip step_state_series_cluster_B (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "stepC_threshold_update" / "stepC_model_summary.csv"]):
            print("[pipeline] stepC_threshold_update")
            stepC_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip stepC_threshold_update (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "stepE_modes" / "stepE_threshold_modes.csv"]):
            print("[pipeline] stepE_threshold_modes")
            stepE_modes.main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip stepE_threshold_modes (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "stepE_event_prepost" / "event_prepost_diff_long.csv"]):
            print("[pipeline] stepE_event_prepost")
            stepE_prepost.main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip stepE_event_prepost (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "stepF_sign_event_linear" / "stepF_event_binary_coeffs.csv"]):
            print("[pipeline] stepF_sign_event_linear")
            stepF_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip stepF_sign_event_linear (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "step2_epsilon_event_embedding" / "epsilon_event_embeddings.csv"]):
            print("[pipeline] step2_epsilon_event_embedding")
            step2_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip step2_epsilon_event_embedding (outputs exist)")

    if "feature_stability" in stages:
        from old.stepG_feature_stability import main as stepG_feat_main

        if _should_run(args.recompute, [paths.derived_dir / "stepG_outputs" / "event_binary_stability.csv"]):
            print("[pipeline] stepG_feature_stability")
            stepG_feat_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip stepG_feature_stability (outputs exist)")

    if "cross" in stages:
        from old.stepG_cluster_stability import main as stepG_cluster_main
        from old.stepH_semantic_cluster_alignment import main as stepH_main
        from old.step_label_lr_proposal import main as step_label_main

        if _should_run(args.recompute, [paths.derived_dir / "stepG_cluster_stability_pairwise.csv"]):
            print("[pipeline] stepG_cluster_stability")
            stepG_cluster_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip stepG_cluster_stability (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "stepH_semantic_alignment" / "aligned_cluster_by_window_with_agreement.csv"]):
            print("[pipeline] stepH_semantic_cluster_alignment")
            stepH_main(["--runs", run_id])
        else:
            print("[pipeline] skip stepH_semantic_cluster_alignment (outputs exist)")

        if _should_run(args.recompute, [paths.derived_dir / "label_lr" / "lr_mult_by_cluster.csv"]):
            print("[pipeline] step_label_lr_proposal")
            step_label_main(["--run-dir", str(paths.run_dir)])
        else:
            print("[pipeline] skip step_label_lr_proposal (outputs exist)")


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train")
    sub.add_parser("detect-critical")
    sub.add_parser("collect-runs")
    sub.add_parser("summarize-p-direction")
    sub.add_parser("pipeline")

    args, rest = ap.parse_known_args(argv)

    if args.cmd == "train":
        from train import main as train_main
        train_main(rest)
        return
    if args.cmd == "detect-critical":
        from detect_critical import main as detect_main
        detect_main(rest)
        return
    if args.cmd == "collect-runs":
        from collect_runs import main as collect_main
        collect_main(rest)
        return
    if args.cmd == "summarize-p-direction":
        from summarize_p_direction import main as summarize_main
        summarize_main(rest)
        return
    if args.cmd == "pipeline":
        run_pipeline(rest)
        return


if __name__ == "__main__":
    main()
