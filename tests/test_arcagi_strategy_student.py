import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples" / "case_studies" / "arcagi_strategy_student"
PUBLIC_RESULTS = CASE / "results" / "visible_arcagi_a_v1"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, CASE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

if NUMPY_AVAILABLE:
    pipeline = load_module("arcagi_strategy_pipeline", "pipeline.py")
    sys.path.insert(0, str(CASE))
    evaluation = load_module("arcagi_strategy_evaluation", "evaluate_student.py")
    ensemble = load_module("arcagi_strategy_ensemble", "evaluate_ensemble.py")
    kaggle_submission = load_module("arcagi_strategy_kaggle_submission", "kaggle_submission.py")
    kaggle_bundle = load_module("arcagi_strategy_kaggle_bundle", "build_kaggle_bundle.py")
    lora_converter = load_module("arcagi_strategy_lora_converter", "convert_mlx_lora_to_peft.py")
    kaggle_merger = load_module("arcagi_strategy_kaggle_merger", "merge_kaggle_submissions.py")
    kaggle_route = load_module("arcagi_strategy_kaggle_route", "run_kaggle_route.py")
    kaggle_scorer = load_module("arcagi_strategy_kaggle_scorer", "score_kaggle_submission.py")
    peft_trainer = load_module("arcagi_strategy_peft_trainer", "train_peft_strategy.py")
else:
    pipeline = None
    evaluation = None
    ensemble = None
    kaggle_submission = None
    kaggle_bundle = None
    lora_converter = None
    kaggle_merger = None
    kaggle_route = None
    kaggle_scorer = None
    peft_trainer = None


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is an optional ARC case-study dependency")
class ArcAgiStrategyStudentTests(unittest.TestCase):
    def test_d4_transforms_cover_eight_distinct_orientations(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        values = {
            tuple(tuple(row) for row in pipeline.d4(grid, name))
            for name in pipeline.D4_NAMES
        }
        self.assertEqual(len(values), 8)

    def test_recolor_requires_and_applies_permutation(self):
        self.assertEqual(pipeline.recolor([[0, 1], [2, 3]], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]), [[9, 8], [7, 6]])
        with self.assertRaises(ValueError):
            pipeline.recolor([[0]], [0] * 10)

    def test_transformed_task_keeps_test_output_out_of_prompt(self):
        task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]], "output": [[4]]}],
        }
        transformed = pipeline.transform_task(task, 0, "identity", list(range(10)), [0])
        prompt = pipeline.prompt_for(transformed)
        self.assertIn('"test_input":[[3]]', prompt)
        self.assertNotIn('"output":[[4]]', prompt)

    def test_strategy_parser_accepts_native_tool_name_and_rejects_unknown(self):
        allowed = {"try_crop", "try_repeat"}
        self.assertEqual(evaluation.parse_strategy('<tool_call>{"name":"arc_try_crop"}</tool_call>', allowed), "try_crop")
        self.assertEqual(
            evaluation.parse_strategy('<tool_call>{"name":"arc_apply_strategy","arguments":{"strategy":"try_repeat"}}</tool_call>', allowed),
            "try_repeat",
        )
        self.assertIsNone(evaluation.parse_strategy("arc_try_invented", allowed))

    def test_router_tool_has_closed_strategy_enum(self):
        tool = pipeline.strategy_tool(["try_crop", "try_repeat"], "a" * 64)
        parameters = tool["function"]["parameters"]
        self.assertEqual(parameters["properties"]["strategy"]["enum"], ["try_crop", "try_repeat"])
        self.assertFalse(parameters["additionalProperties"])

    def test_ensemble_deduplicates_and_honors_prediction_cap(self):
        import numpy as np

        values = []
        ensemble.append_unique(values, np.array([[1]]), 2)
        ensemble.append_unique(values, np.array([[1]]), 2)
        ensemble.append_unique(values, np.array([[2]]), 2)
        ensemble.append_unique(values, np.array([[3]]), 2)
        self.assertEqual([value.tolist() for value in values], [[[1]], [[2]]])

    def test_kaggle_challenges_reject_test_outputs(self):
        challenge = {
            "task": {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": [[4]]}],
            }
        }
        with self.assertRaisesRegex(ValueError, "test outputs are forbidden"):
            kaggle_submission.validate_label_blind_challenges(challenge)

    def test_kaggle_submission_emits_exactly_two_attempts_without_labels(self):
        import numpy as np

        challenges = kaggle_submission.validate_label_blind_challenges(
            {
                "task": {
                    "train": [{"input": [[1]], "output": [[2]]}],
                    "test": [{"input": [[3]]}],
                }
            }
        )

        def predict(*_args):
            return kaggle_submission.PredictionResult(
                guesses=[np.array([[7]])],
                selected_strategy="try_crop",
                raw_output_sha256="a" * 64,
            )

        submission, audit = kaggle_submission.build_submission(challenges, predict)
        self.assertEqual(
            submission,
            {"task": [{"attempt_1": [[7]], "attempt_2": [[7]]}]},
        )
        self.assertEqual(audit[0]["fallback"], "duplicate_only_valid_prediction")
        self.assertNotIn("task", audit[0]["task_ref_sha256"])

    def test_mlx_to_peft_key_contract_transposes_complete_pairs(self):
        keys = [
            f"model.layers.27.self_attn.q_proj.lora_{side}"
            for side in ("a", "b")
        ]
        contract = lora_converter.conversion_contract(
            {
                "num_layers": 1,
                "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
            },
            keys,
        )
        self.assertEqual(contract["lora_alpha"], 160.0)
        self.assertEqual(contract["layers"], [27])
        self.assertEqual(contract["target_modules"], ["q_proj"])
        self.assertEqual(
            lora_converter.peft_key(keys[0])[0],
            "base_model.model.model.layers.27.self_attn.q_proj.lora_A.weight",
        )

    def test_kaggle_scorer_uses_pass_at_two_exact_grid_matching(self):
        aggregate, outcomes = kaggle_scorer.score_submission(
            {"task": [{"attempt_1": [[0]], "attempt_2": [[7]]}]},
            {"task": [[[7]]]},
        )
        self.assertEqual(aggregate, {"passed": 1, "total": 1, "exact_rate": 1.0})
        self.assertEqual(outcomes[0]["matched_attempt"], 2)

    def test_kaggle_merge_prefers_unique_candidates_in_checkpoint_order(self):
        primary = {"task": [{"attempt_1": [[1]], "attempt_2": [[1]]}]}
        secondary = {"task": [{"attempt_1": [[2]], "attempt_2": [[3]]}]}
        merged, audit = kaggle_merger.merge_submissions(primary, secondary)
        self.assertEqual(merged, {"task": [{"attempt_1": [[1]], "attempt_2": [[2]]}]})
        self.assertFalse(audit[0]["duplicated_only_candidate"])

    def test_kaggle_merge_rejects_mismatched_task_sets(self):
        with self.assertRaisesRegex(ValueError, "identifiers"):
            kaggle_merger.merge_submissions(
                {"one": [{"attempt_1": [[1]], "attempt_2": [[1]]}]},
                {"two": [{"attempt_1": [[1]], "attempt_2": [[1]]}]},
            )

    def test_kaggle_route_gate_fails_closed_on_accuracy_or_router_errors(self):
        passed, failures = kaggle_route.visible_gate(
            aggregate={"passed": 51, "total": 52},
            router_receipts=[
                {"invalid_strategy_count": 0, "error_count": 0},
                {"invalid_strategy_count": 1, "error_count": 0},
            ],
            expected_total=52,
            minimum_passed=52,
        )
        self.assertFalse(passed)
        self.assertEqual(
            failures,
            ["visible_accuracy_below_threshold", "router_2_invalid_strategy"],
        )

    def test_kaggle_route_gate_accepts_clean_exact_visible_parity(self):
        passed, failures = kaggle_route.visible_gate(
            aggregate={"passed": 52, "total": 52},
            router_receipts=[
                {"invalid_strategy_count": 0, "error_count": 0},
                {"invalid_strategy_count": 0, "error_count": 0},
            ],
            expected_total=52,
            minimum_passed=52,
        )
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_kaggle_bundle_requires_normalized_private_ids(self):
        self.assertEqual(
            kaggle_bundle.validate_kaggle_id("owner/hfr-arcagi-private-v1"),
            "owner/hfr-arcagi-private-v1",
        )
        with self.assertRaises(ValueError):
            kaggle_bundle.validate_kaggle_id("OWNER/Has Spaces")

    def test_kaggle_bundle_copies_without_aliasing_source_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            destination = root / "bundle/copied.bin"
            source.write_bytes(b"reviewed source")
            source.chmod(0o644)
            kaggle_bundle.copy_file(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertNotEqual(destination.stat().st_ino, source.stat().st_ino)
            self.assertEqual(source.stat().st_mode & 0o777, 0o644)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

    def test_kaggle_bundle_verifies_exact_pinned_file_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "file.bin").write_bytes(b"reviewed")
            expected = {"file.bin": hashlib.sha256(b"reviewed").hexdigest()}
            kaggle_bundle.verify_exact_file_set(root, expected, "fixture")
            (root / "extra.bin").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "reviewed pinned set"):
                kaggle_bundle.verify_exact_file_set(root, expected, "fixture")

    def test_kaggle_notebook_installs_only_pinned_offline_wheels(self):
        source = "".join(kaggle_bundle.notebook_payload()["cells"][0]["source"])
        compile(source, "arcagi_kaggle_route.ipynb", "exec")
        self.assertIn("--no-index", source)
        self.assertIn("--no-deps", source)
        self.assertNotIn("https://", source)

    def test_kaggle_kernel_metadata_is_private_gpu_offline(self):
        metadata = kaggle_bundle.kernel_metadata_payload(
            dataset_id="owner/private-data",
            kernel_id="owner/private-kernel",
            competition="arc-prize-2026-arc-agi-2",
            notebook_name="route.ipynb",
        )
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["enable_gpu"], "true")
        self.assertEqual(metadata["enable_tpu"], "false")
        self.assertEqual(metadata["enable_internet"], "false")
        self.assertEqual(metadata["dataset_sources"], ["owner/private-data"])
        self.assertEqual(metadata["competition_sources"], ["arc-prize-2026-arc-agi-2"])

    def test_peft_training_masks_the_prompt_and_keeps_tool_call_supervision(self):
        class Tokenizer:
            @staticmethod
            def apply_chat_template(messages, **_kwargs):
                return "prompt" if len(messages) == 2 else "promptcompletion"

            @staticmethod
            def encode(value, add_special_tokens=False):
                self.assertFalse(add_special_tokens)
                return [ord(character) for character in value]

        row = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "tool"}}]},
            ],
            "tools": [{"type": "function"}],
        }
        input_ids, labels = peft_trainer.render_training_row(Tokenizer(), row)
        self.assertEqual(len(input_ids), len(labels))
        self.assertEqual(labels[: len("prompt")], [-100] * len("prompt"))
        self.assertEqual(labels[len("prompt") :], input_ids[len("prompt") :])


class ArcAgiPublicEvidenceTests(unittest.TestCase):
    def test_public_aggregate_csv_matches_canonical_metrics(self):
        metrics = json.loads((PUBLIC_RESULTS / "metrics.json").read_text(encoding="utf-8"))
        with (PUBLIC_RESULTS / "metrics_summary.csv").open(encoding="utf-8", newline="") as handle:
            rows = {row["configuration"]: row for row in csv.DictReader(handle)}

        self.assertEqual(set(rows), set(metrics["results"]))
        for configuration, result in metrics["results"].items():
            row = rows[configuration]
            strategy_count = result.get("exact_strategy_count", result.get("strategy_recall_count"))
            self.assertEqual(int(row["exact_strategy_or_recall_count"]), strategy_count)
            self.assertEqual(int(row["exact_grid_count"]), result["exact_grid_count"])
            self.assertEqual(int(row["example_count"]), result["example_count"])
            self.assertEqual(float(row["exact_grid_rate"]), result["exact_grid_rate"])
            self.assertEqual(int(row["invalid_action_count"]), result["invalid_action_count"])

    def test_public_checksum_manifest_verifies(self):
        checksums = {}
        for line in (PUBLIC_RESULTS / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, filename = line.split("  ", 1)
            checksums[filename] = digest

        expected_files = {"metrics.json", "metrics_summary.csv", "outcomes.jsonl"}
        self.assertEqual(set(checksums), expected_files)
        for filename, expected_digest in checksums.items():
            actual_digest = hashlib.sha256((PUBLIC_RESULTS / filename).read_bytes()).hexdigest()
            self.assertEqual(actual_digest, expected_digest)

    def test_public_evidence_replays_from_anonymous_outcomes(self):
        metrics = json.loads((PUBLIC_RESULTS / "metrics.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (PUBLIC_RESULTS / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertFalse(metrics["benchmark_claim_allowed"])
        self.assertEqual([row["query_index"] for row in rows], list(range(1, 53)))
        expected_fields = {
            "base",
            "base_invalid_action",
            "ensemble",
            "lora_0900",
            "lora_0900_invalid_action",
            "lora_1200",
            "lora_1200_invalid_action",
            "query_index",
        }
        self.assertTrue(all(set(row) == expected_fields for row in rows))

        results = metrics["results"]
        self.assertEqual(sum(row["base"] for row in rows), results["base"]["exact_grid_count"])
        self.assertEqual(sum(row["lora_0900"] for row in rows), results["lora_checkpoint_0900"]["exact_grid_count"])
        self.assertEqual(sum(row["lora_1200"] for row in rows), results["lora_checkpoint_1200"]["exact_grid_count"])
        self.assertEqual(sum(row["ensemble"] for row in rows), results["two_checkpoint_ensemble"]["exact_grid_count"])
        self.assertEqual(sum(row["base_invalid_action"] for row in rows), results["base"]["invalid_action_count"])
        self.assertEqual(sum(row["lora_0900_invalid_action"] for row in rows), 0)
        self.assertEqual(sum(row["lora_1200_invalid_action"] for row in rows), 0)

        serialized = json.dumps(rows, sort_keys=True)
        for forbidden in ("episode_id", "task_family", "raw_output", "selected_strategy", "/Users/"):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
