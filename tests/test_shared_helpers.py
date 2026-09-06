from __future__ import annotations

import hashlib
import importlib
import tempfile
import unittest
from pathlib import Path
from typing import Callable


_HASH_HELPERS = (
    ("flightrecorder.action_ledger", "_sha256"),
    ("flightrecorder.agentic_loop_governance", "_sha256"),
    ("flightrecorder.agentic_loop_ledger", "_sha256"),
    ("flightrecorder.agentic_training_loop_plan", "_sha256"),
    ("flightrecorder.agentic_training_plan", "_sha256"),
    ("flightrecorder.atomic_json", "_sha256"),
    ("flightrecorder.bundle", "_sha256"),
    ("flightrecorder.cli", "_sha256_file"),
    ("flightrecorder.cloud_training", "_sha256"),
    ("flightrecorder.data_governance", "_sha256_file"),
    ("flightrecorder.dataset_curation", "_sha256"),
    ("flightrecorder.decision_gate", "_sha256"),
    ("flightrecorder.eval_summary", "_sha256"),
    ("flightrecorder.external_eval", "_sha256"),
    ("flightrecorder.governance", "_sha256"),
    ("flightrecorder.harness", "_sha256"),
    ("flightrecorder.heldout_manifest", "_sha256_file"),
    ("flightrecorder.improvement_ledger", "_sha256"),
    ("flightrecorder.improvement_plan", "_sha256"),
    ("flightrecorder.lineage", "_sha256"),
    ("flightrecorder.lora_recipe_search", "_file_sha256"),
    ("flightrecorder.mlx_prefix_equivalence_smoke", "_sha256_file"),
    ("flightrecorder.model_grader", "_sha256"),
    ("flightrecorder.model_registry", "_sha256"),
    ("flightrecorder.next_iteration_schedule", "_sha256"),
    ("flightrecorder.path_safety", "_sha256_file"),
    ("flightrecorder.preflight", "_sha256"),
    ("flightrecorder.promotion_archive", "_sha256"),
    ("flightrecorder.promotion_ledger", "_sha256"),
    ("flightrecorder.rejection_sampling", "_sha256"),
    ("flightrecorder.repair", "_sha256"),
    ("flightrecorder.review", "_sha256_file"),
    ("flightrecorder.rollout_generation", "_sha256"),
    ("flightrecorder.state_capture", "_sha256"),
    ("flightrecorder.tau3_candidate_attempts", "_sha256_file"),
    ("flightrecorder.tau3_candidate_identity", "_sha256_file"),
    ("flightrecorder.tau3_competitive_v3", "_sha256_file"),
    ("flightrecorder.tau3_competitive_v3_training_evidence", "_sha256_file"),
    ("flightrecorder.tau3_competitive_v3_training_stage", "_sha256_file"),
    ("flightrecorder.tau3_conversation_ingest", "_file_sha256"),
    ("flightrecorder.tau3_development_evaluation", "_sha256_file"),
    ("flightrecorder.tau3_development_screening", "_sha256_file"),
    ("flightrecorder.tau3_execution_bundle", "_sha256_file"),
    ("flightrecorder.tau3_execution_validation", "_sha256_file"),
    ("flightrecorder.tau3_exposure", "_sha256_file"),
    ("flightrecorder.tau3_generation_retry", "_sha256"),
    ("flightrecorder.tau3_internal_validation", "_sha256_file"),
    ("flightrecorder.tau3_mlx_training", "_sha256_file"),
    ("flightrecorder.tau3_policy_complete_dataset", "_sha256"),
    ("flightrecorder.tau3_prefix_equivalence", "_sha256_file"),
    ("flightrecorder.tau3_prefix_equivalence_sample", "_sha256_file"),
    ("flightrecorder.tau3_protocol_freeze", "_sha256_file"),
    ("flightrecorder.tau3_sealed_grid_completeness", "_sha256_file"),
    ("flightrecorder.tau3_source_partition", "_sha256_file"),
    ("flightrecorder.tau3_training_mixture", "_sha256"),
    ("flightrecorder.tau3_v3_scenarios", "_sha256_file"),
    ("flightrecorder.trainer_archive", "_sha256"),
    ("flightrecorder.trainer_archive_check", "_sha256"),
    ("flightrecorder.trainer_consumer_plan", "_sha256"),
    ("flightrecorder.training", "_sha256_file"),
    ("flightrecorder.validation", "_sha256"),
)


def _hash_helper(module_name: str, function_name: str) -> Callable[[Path], str]:
    module = importlib.import_module(module_name)
    helper = getattr(module, function_name)
    return helper


class SharedHashHelperCharacterizationTests(unittest.TestCase):
    def test_all_existing_aliases_hash_binary_content(self) -> None:
        payload = b"\x00\xffflight-recorder\x80\x00"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(payload)

            for module_name, function_name in _HASH_HELPERS:
                with self.subTest(module=module_name, function=function_name):
                    self.assertEqual(_hash_helper(module_name, function_name)(path), expected)

    def test_existing_hash_behavior_covers_empty_and_multiple_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            helper = _hash_helper("flightrecorder.atomic_json", "_sha256")
            for payload in (b"", b"a" * (2 * 1024 * 1024 + 17)):
                with self.subTest(size=len(payload)):
                    path.write_bytes(payload)
                    self.assertEqual(helper(path), hashlib.sha256(payload).hexdigest())

    def test_existing_hash_behavior_propagates_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = _hash_helper("flightrecorder.atomic_json", "_sha256")
            with self.assertRaises(FileNotFoundError):
                helper(root / "missing.bin")
            with self.assertRaises(IsADirectoryError):
                helper(root)


class SharedGateMetricsCharacterizationTests(unittest.TestCase):
    def test_missing_summary_has_unavailable_zero_metrics(self) -> None:
        expected = {
            "available": False,
            "passed": False,
            "strict": False,
            "target_count": 0,
            "error_count": 0,
            "warning_count": 0,
        }
        for module_name in (
            "flightrecorder.calibration",
            "flightrecorder.compare_gate",
            "flightrecorder.training_gate",
            "flightrecorder.reviewed_gate",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(module._validation_metrics(None), expected)

    def test_bool_flags_and_count_types_preserve_existing_coercion(self) -> None:
        summary = {
            "passed": "yes",
            "strict": [],
            "target_count": True,
            "error_count": 4,
            "warning_count": "5",
        }
        expected = {
            "available": True,
            "passed": True,
            "strict": False,
            "target_count": 0,
            "error_count": 4,
            "warning_count": 0,
        }
        for module_name in (
            "flightrecorder.calibration",
            "flightrecorder.compare_gate",
            "flightrecorder.training_gate",
            "flightrecorder.reviewed_gate",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(module._validation_metrics(summary), expected)

    def test_negative_counts_distinguish_reviewed_gate(self) -> None:
        summary = {"target_count": -1, "error_count": -2, "warning_count": -3}
        preserving = {
            "available": True,
            "passed": False,
            "strict": False,
            "target_count": -1,
            "error_count": -2,
            "warning_count": -3,
        }
        for module_name in (
            "flightrecorder.calibration",
            "flightrecorder.compare_gate",
            "flightrecorder.training_gate",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(module._validation_metrics(summary), preserving)

        reviewed_gate = importlib.import_module("flightrecorder.reviewed_gate")
        self.assertEqual(
            reviewed_gate._validation_metrics(summary),
            {
                **preserving,
                "target_count": 0,
                "error_count": 0,
                "warning_count": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
