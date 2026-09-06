"""CLI commands for the exports domain."""

from __future__ import annotations

import argparse
from ..training import TrainingExportError, export_compare_rl_dataset, export_rl_dataset
from .shared import _metadata_arg, _metadata_options


def cmd_export_rl(args: argparse.Namespace) -> int:
    metadata = _metadata_options(args.metadata)
    manifest = export_rl_dataset(
        args.runs,
        args.out,
        reward_scale=args.reward_scale,
        min_score_gap=args.min_score_gap,
        max_pairs_per_family=args.max_pairs_per_family,
        preserve_paths=args.preserve_paths,
        metadata=metadata,
    )
    print(
        "wrote RL export "
        f"episodes={manifest['episode_count']} rewards={manifest['reward_count']} "
        f"step_rewards={manifest['step_reward_count']} "
        f"preferences={manifest['preference_count']} failure_modes={manifest['failure_mode_count']} "
        f"sft={manifest['sft_count']} dpo={manifest['dpo_count']} "
        f"reward_model={manifest['reward_model_count']} "
        f"quality_flags={manifest['quality_flag_count']} out={args.out}"
    )
    return 0


def cmd_export_compare_rl(args: argparse.Namespace) -> int:
    metadata = _metadata_options(args.metadata)
    manifest = export_compare_rl_dataset(
        args.baseline,
        args.candidate,
        args.out,
        reward_scale=args.reward_scale,
        min_score_gap=args.min_score_gap,
        contract_scope=args.contract_scope,
        preserve_paths=args.preserve_paths,
        metadata=metadata,
    )
    print(
        "wrote compare RL export "
        f"pairs={manifest['pair_count']} dpo={manifest['dpo_count']} "
        f"candidate_wins={manifest['candidate_win_count']} "
        f"baseline_wins={manifest['baseline_win_count']} out={args.out}"
    )
    return 0


def register_exports_1(subparsers: argparse._SubParsersAction) -> None:
    export_rl = subparsers.add_parser("export-rl", help="Export completed runs as future RL training artifacts")
    export_rl.add_argument("--runs", required=True, help="Directory containing Flight Recorder run subdirectories")
    export_rl.add_argument("--out", required=True, help="Output directory for evidence and trainer-ready artifacts")
    export_rl.add_argument(
        "--reward-scale",
        default="score",
        choices=["score", "binary", "signed"],
        help="Reward transform: score=0..1, binary=pass/fail, signed=-1..1",
    )
    export_rl.add_argument("--min-score-gap", type=int, default=1, help="Minimum score gap for a preference pair")
    export_rl.add_argument(
        "--max-pairs-per-family",
        type=int,
        default=0,
        help="Maximum preference pairs per task family; 0 means unlimited",
    )
    export_rl.add_argument("--preserve-paths", action="store_true", help="Allow absolute source/output paths in exported metadata")
    export_rl.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach experiment metadata to manifest, dataset metrics, and dataset card; may be repeated",
    )
    export_rl.set_defaults(func=cmd_export_rl)

    export_compare_rl = subparsers.add_parser(
        "export-compare-rl",
        help="Export paired baseline/candidate runs as preference artifacts",
    )
    export_compare_rl.add_argument("--baseline", required=True, help="Baseline run-suite directory")
    export_compare_rl.add_argument("--candidate", required=True, help="Candidate run-suite directory")
    export_compare_rl.add_argument("--out", required=True, help="Output directory for comparison preference artifacts")
    export_compare_rl.add_argument(
        "--reward-scale",
        default="score",
        choices=["score", "binary", "signed"],
        help="Reward transform used inside episode views: score=0..1, binary=pass/fail, signed=-1..1",
    )
    export_compare_rl.add_argument("--min-score-gap", type=int, default=1, help="Minimum absolute score gap for an improvement pair")
    export_compare_rl.add_argument(
        "--contract-scope",
        default="scenario",
        choices=["scenario", "scenario-and-trace"],
        help="Fingerprint contract to compare: scenario for live improvement loops, scenario-and-trace for strict fixture replay",
    )
    export_compare_rl.add_argument("--preserve-paths", action="store_true", help="Allow absolute source/output paths in exported metadata")
    export_compare_rl.add_argument(
        "--metadata",
        action="append",
        default=[],
        type=_metadata_arg,
        metavar="KEY=VALUE",
        help="Attach experiment metadata to the comparison export; may be repeated",
    )
    export_compare_rl.set_defaults(func=cmd_export_compare_rl)

