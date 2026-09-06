"""CLI commands for the constants domain."""

from __future__ import annotations

import re


RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"


GOAL3_HANDOFF_SCHEMA_VERSION = "hfr.goal3_handoff.v1"


FAMILY_SUFFIX_RE = re.compile(r"([_-](good|bad|pass|fail|passing|failing|chosen|rejected))+$", re.IGNORECASE)


TRACE_FORMAT_CHOICES = [
    "auto",
    "trajectory_jsonl",
    "observer_jsonl",
    "openclaw_jsonl",
    "coven_jsonl",
    "atof_jsonl",
    "atif_json",
    "normalized_json",
]


_OPTIONAL_RUN_ARTIFACTS = (
    "trajectory_v2.json",
    "before_state_snapshot.json",
    "state_snapshot.json",
    "state_diff.json",
    "regression_scenario.json",
    "raw_trace.sensitive.json",
)


_OPTIONAL_RUN_ARTIFACT_RECORDS = {
    "trajectory_v2": "trajectory_v2.json",
    "before_state_snapshot": "before_state_snapshot.json",
    "state_snapshot": "state_snapshot.json",
    "state_diff": "state_diff.json",
    "regression_scenario": "regression_scenario.json",
    "raw_trace_sensitive": "raw_trace.sensitive.json",
}


_REQUIRED_RUN_ARTIFACTS = {
    "normalized_trace": "normalized_trace.json",
    "scorecard": "scorecard.json",
    "task_completion": "task_completion.json",
    "run_digest": "run_digest.json",
    "report": "report.html",
}


_RUN_LINEAGE_OUTPUT_NAMES = frozenset(
    {*_REQUIRED_RUN_ARTIFACTS, *_OPTIONAL_RUN_ARTIFACT_RECORDS, "junit", "markdown"}
)
