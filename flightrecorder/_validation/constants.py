"""Extracted validation implementation."""

from __future__ import annotations


VALIDATION_SCHEMA_VERSION = "hfr.validation.v1"

RUN_SUITE_SCHEMA_VERSION = "hfr.run_suite.v1"

EVAL_SUITE_MANIFEST_SCHEMA_VERSION = "hfr.eval_suite_manifest.v1"

LEGACY_LIVE_SMOKE_SUMMARY_SCHEMA_VERSIONS = {"hfr.live_smoke.summary.v1"}

TRAINER_WRAPPER_DRY_RUN_SCHEMA_VERSION = "hfr.example_trainer_wrapper_dry_run.v1"

HARNESS_REPLAY_RESULT_SCHEMA_VERSION = "hfr.harness_replay_result.v1"

HARNESS_SUITE_RESULT_SCHEMA_VERSION = "hfr.harness_suite_result.v1"

RUN_SUITE_HARNESS_SOURCE = "flightrecorder run-suite --evidence-handoff"

SERVING_PROFILE_SCHEMA_VERSION = "hfr.serving_profile.v1"

SERVING_COMPATIBILITY_REPORT_SCHEMA_VERSION = "hfr.serving_compatibility_report.v1"

SERVING_ENDPOINT_CHECK_SCHEMA_VERSION = "hfr.serving_endpoint_check.v1"

SERVING_LIFECYCLE_SCHEMA_VERSION = "hfr.serving_lifecycle.v1"

SERVING_DEMO_RUN_SCHEMA_VERSION = "hfr.serving_demo_run.v1"

SERVING_CAPABILITY_STATUSES = {"supported", "not_verified"}

SERVING_LIFECYCLE_PREFLIGHT_ARTIFACTS = ("serving_profile", "compatibility_report", "serving_check")

_COMPANION_NOT_PROVIDED = object()
