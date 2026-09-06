"""Composition root for the Flight Recorder argument parser."""

from __future__ import annotations

import argparse
from .artifacts import register_artifacts_1, register_artifacts_2
from .basic import register_basic_1
from .evals import register_evals_1, register_evals_2, register_evals_3
from .evidence import register_evidence_1, register_evidence_2
from .exports import register_exports_1
from .gates import register_gates_1, register_gates_2, register_gates_3
from .generation import register_generation_1, register_generation_2
from .governance import register_governance_1, register_governance_2, register_governance_3, register_governance_4, register_governance_5
from .loop import register_loop_1, register_loop_2
from .models import register_models_1
from .run_commands import register_run_commands_1
from .suite import register_suite_1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flightrecorder", description="Hermes Autonomy Flight Recorder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_basic_1(subparsers)
    register_run_commands_1(subparsers)
    register_suite_1(subparsers)
    register_artifacts_1(subparsers)
    register_models_1(subparsers)
    register_generation_1(subparsers)
    register_evals_1(subparsers)
    register_loop_1(subparsers)
    register_evals_2(subparsers)
    register_governance_1(subparsers)
    register_loop_2(subparsers)
    register_generation_2(subparsers)
    register_evals_3(subparsers)
    register_evidence_1(subparsers)
    register_gates_1(subparsers)
    register_evidence_2(subparsers)
    register_governance_2(subparsers)
    register_gates_2(subparsers)
    register_governance_3(subparsers)
    register_gates_3(subparsers)
    register_governance_4(subparsers)
    register_exports_1(subparsers)
    register_governance_5(subparsers)
    register_artifacts_2(subparsers)
    return parser
