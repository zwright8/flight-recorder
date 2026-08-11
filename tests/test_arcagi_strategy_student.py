import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples" / "case_studies" / "arcagi_strategy_student"
PUBLIC_RESULTS = CASE / "results" / "visible_arcagi_a_v1"
KAGGLE_PUBLIC_RESULTS = CASE / "results" / "kaggle_arcagi2_router_only_v1"


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
    kaggle_runtime_overlay = load_module(
        "arcagi_strategy_kaggle_runtime_overlay",
        "build_kaggle_runtime_overlay.py",
    )
    lora_converter = load_module("arcagi_strategy_lora_converter", "convert_mlx_lora_to_peft.py")
    kaggle_merger = load_module("arcagi_strategy_kaggle_merger", "merge_kaggle_submissions.py")
    kaggle_comparison = load_module(
        "arcagi_strategy_kaggle_comparison",
        "run_kaggle_comparison_route.py",
    )
    kaggle_route = load_module("arcagi_strategy_kaggle_route", "run_kaggle_route.py")
    kaggle_scorer = load_module("arcagi_strategy_kaggle_scorer", "score_kaggle_submission.py")
    peft_trainer = load_module("arcagi_strategy_peft_trainer", "train_peft_strategy.py")
else:
    pipeline = None
    evaluation = None
    ensemble = None
    kaggle_submission = None
    kaggle_bundle = None
    kaggle_runtime_overlay = None
    lora_converter = None
    kaggle_merger = None
    kaggle_comparison = None
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

    def test_kaggle_submission_reports_each_completed_input(self):
        import numpy as np

        challenges = kaggle_submission.validate_label_blind_challenges(
            {
                "task": {
                    "train": [{"input": [[1]], "output": [[2]]}],
                    "test": [{"input": [[3]]}, {"input": [[4]]}],
                }
            }
        )
        progress = []

        def predict(*_args):
            return kaggle_submission.PredictionResult(
                guesses=[np.array([[7]])],
                selected_strategy="try_crop",
                raw_output_sha256="a" * 64,
            )

        _, audit = kaggle_submission.build_submission(
            challenges,
            predict,
            max_seconds=60,
            progress=progress.append,
        )
        self.assertEqual(len(audit), 2)
        self.assertEqual([value["completed"] for value in progress], [1, 2])
        self.assertEqual([value["total"] for value in progress], [2, 2])

    def test_kaggle_candidate_policy_adds_distinct_teacher_fallback_with_provenance(self):
        import numpy as np

        class Data:
            @staticmethod
            def data():
                return np.array([[3]])

        class TestSet:
            @staticmethod
            def get_input_data():
                return Data()

        class Problem:
            @staticmethod
            def training_set():
                return []

            @staticmethod
            def test_set():
                return TestSet()

        class Executor:
            @staticmethod
            def as_guess_list(value):
                return value

            @staticmethod
            def is_valid_grid(value):
                return np.asarray(value).ndim == 2

            @staticmethod
            def try_crop(_training, _test_input):
                return [np.array([[7]])]

            @staticmethod
            def make_predictions(_problem):
                return [np.array([[8]])]

        guesses, sources, error = kaggle_submission.execute_candidate_policy(
            Executor(), Problem(), "try_crop", "router-then-teacher"
        )
        self.assertEqual([value.tolist() for value in guesses], [[[7]], [[8]]])
        self.assertEqual(
            sources,
            ("router_selected_strategy", "deterministic_teacher_fallback"),
        )
        self.assertIsNone(error)

        router_only, router_sources, _ = kaggle_submission.execute_candidate_policy(
            Executor(), Problem(), "try_crop", "router-only"
        )
        self.assertEqual([value.tolist() for value in router_only], [[[7]]])
        self.assertEqual(router_sources, ("router_selected_strategy",))

    def test_kaggle_submission_fails_closed_when_time_bound_expires(self):
        import numpy as np

        challenges = kaggle_submission.validate_label_blind_challenges(
            {
                "task": {
                    "train": [{"input": [[1]], "output": [[2]]}],
                    "test": [{"input": [[3]]}],
                }
            }
        )
        original_monotonic = kaggle_submission.time.monotonic
        timeline = iter((0.0, 2.0))
        kaggle_submission.time.monotonic = lambda: next(timeline)
        try:
            with self.assertRaisesRegex(TimeoutError, "0/1 test inputs"):
                kaggle_submission.build_submission(
                    challenges,
                    lambda *_args: kaggle_submission.PredictionResult(
                        guesses=[np.array([[7]])],
                        selected_strategy="try_crop",
                        raw_output_sha256="a" * 64,
                    ),
                    max_seconds=1,
                )
        finally:
            kaggle_submission.time.monotonic = original_monotonic

    def test_hf_router_loads_exact_adapter_state_without_peft_reload(self):
        source = (CASE / "kaggle_submission.py").read_text(encoding="utf-8")
        self.assertIn("get_peft_model_state_dict", source)
        self.assertIn("set_peft_model_state_dict", source)
        self.assertIn("topology_then_exact_state_dict", source)
        self.assertNotIn("PeftModel.from_pretrained", source)

    def test_hf_router_toggles_only_a_resident_adapter(self):
        router = kaggle_submission.HfStrategyRouter.__new__(
            kaggle_submission.HfStrategyRouter
        )
        router.adapter_resident = True
        router.adapter_enabled = True
        router.set_adapter_enabled(False)
        self.assertFalse(router.adapter_enabled)
        router.set_adapter_enabled(True)
        self.assertTrue(router.adapter_enabled)

        router.adapter_resident = False
        router.adapter_enabled = False
        with self.assertRaisesRegex(ValueError, "not resident"):
            router.set_adapter_enabled(True)

    def test_kaggle_comparison_loads_one_router_and_toggles_controlled_arms(self):
        source = (CASE / "run_kaggle_comparison_route.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("kaggle_submission.HfStrategyRouter("), 1)
        self.assertIn('shared_router.set_adapter_enabled(arm == "lora")', source)
        self.assertIn('"load_once_for_all_arms": True', source)

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

    def test_kaggle_route_rejects_p100_and_accepts_t4_capability(self):
        with self.assertRaisesRegex(ValueError, "below the required 7.0"):
            kaggle_route.require_supported_cuda_capability((6, 0))
        self.assertEqual(
            kaggle_route.require_supported_cuda_capability((7, 5)),
            (7, 5),
        )

    def test_kaggle_comparison_verifies_exact_frozen_adapter_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp)
            config = adapter / "adapter_config.json"
            weights = adapter / "adapter_model.safetensors"
            config.write_bytes(b"reviewed config")
            weights.write_bytes(b"reviewed weights")
            receipt = {
                "status": "succeeded",
                "target_steps": 900,
                "completed_steps": 900,
                "outputs": {
                    "adapter_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "adapter_model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                },
            }
            (adapter / "training_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            self.assertEqual(kaggle_comparison.verify_frozen_adapter(adapter), receipt)
            weights.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hashes"):
                kaggle_comparison.verify_frozen_adapter(adapter)

    def test_kaggle_comparison_gate_records_invalid_base_action_but_fails_on_execution(self):
        receipt = {
            "task_count": 52,
            "invalid_strategy_count": 1,
            "error_count": 0,
            "inference": {"candidate_policy": "router-only"},
        }
        passed, failures = kaggle_comparison.comparison_visible_gate(
            aggregate={"passed": 52, "total": 52},
            router_receipt=receipt,
            expected_total=52,
            minimum_passed=52,
        )
        self.assertTrue(passed)
        self.assertEqual(failures, [])
        receipt["error_count"] = 1
        passed, failures = kaggle_comparison.comparison_visible_gate(
            aggregate={"passed": 52, "total": 52},
            router_receipt=receipt,
            expected_total=52,
            minimum_passed=52,
        )
        self.assertFalse(passed)
        self.assertEqual(failures, ["router_execution_error"])

    def test_kaggle_comparison_accepts_zero_accuracy_as_an_integrity_clean_baseline(self):
        passed, failures = kaggle_comparison.comparison_visible_gate(
            aggregate={"passed": 0, "total": 52},
            router_receipt={
                "task_count": 52,
                "invalid_strategy_count": 1,
                "error_count": 0,
                "inference": {"candidate_policy": "router-only"},
            },
            expected_total=52,
            minimum_passed=0,
        )
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_kaggle_hidden_comparison_gate_checks_state_before_promotion(self):
        receipt = {
            "task_count": 120,
            "test_input_count": 259,
            "invalid_strategy_count": 2,
            "error_count": 0,
            "inference": {"candidate_policy": "router-only"},
            "adapter": {"enabled": True},
        }
        passed, failures = kaggle_comparison.comparison_hidden_gate(
            router_receipt=receipt,
            expected_task_count=120,
            expected_test_input_count=259,
            expected_adapter_enabled=True,
        )
        self.assertTrue(passed)
        self.assertEqual(failures, [])
        receipt["adapter"]["enabled"] = False
        receipt["error_count"] = 1
        passed, failures = kaggle_comparison.comparison_hidden_gate(
            router_receipt=receipt,
            expected_task_count=120,
            expected_test_input_count=259,
            expected_adapter_enabled=True,
        )
        self.assertFalse(passed)
        self.assertEqual(
            failures,
            ["hidden_router_execution_error", "hidden_adapter_state_mismatch"],
        )

    def test_peft_training_disables_only_incompatible_optional_torchao(self):
        def available():
            return True

        import_utils = SimpleNamespace(is_torchao_available=available)
        torchao_dispatcher = SimpleNamespace(is_torchao_available=available)
        self.assertTrue(
            peft_trainer.disable_incompatible_torchao_dispatcher(
                import_utils,
                torchao_dispatcher,
                "0.10.0",
            )
        )
        self.assertFalse(import_utils.is_torchao_available())
        self.assertFalse(torchao_dispatcher.is_torchao_available())

        supported_import_utils = SimpleNamespace(is_torchao_available=available)
        supported_dispatcher = SimpleNamespace(is_torchao_available=available)
        self.assertFalse(
            peft_trainer.disable_incompatible_torchao_dispatcher(
                supported_import_utils,
                supported_dispatcher,
                "0.16.0",
            )
        )
        self.assertTrue(supported_import_utils.is_torchao_available())
        self.assertTrue(supported_dispatcher.is_torchao_available())

    def test_peft_completion_loss_window_keeps_full_context_but_only_suffix_logits(self):
        logits_to_keep, shifted_labels = peft_trainer.completion_loss_window(
            [-100, -100, -100, 41, 42, 43]
        )
        self.assertEqual(logits_to_keep, 4)
        self.assertEqual(shifted_labels, [41, 42, 43, -100])

        with self.assertRaisesRegex(ValueError, "contiguous suffix"):
            peft_trainer.completion_loss_window([-100, 41, -100, 43])

    def test_peft_training_result_status_fails_closed_on_short_run(self):
        self.assertEqual(peft_trainer.training_result_status(900, 900), "succeeded")
        self.assertEqual(
            peft_trainer.training_result_status(263, 900),
            "time_bound_exhausted",
        )
        with self.assertRaisesRegex(ValueError, "invalid training completion counts"):
            peft_trainer.training_result_status(901, 900)

    def test_peft_continuation_plan_resets_phase_state_without_mutating_base(self):
        base = {"configuration": {"seed": 17}, "data": {"retained_rows": 1247}}
        phase = peft_trainer.training_phase_plan(
            base,
            max_steps=300,
            max_training_seconds=7200,
            initial_adapter_sha256="a" * 64,
            reuse_loaded_model=True,
        )
        self.assertEqual(base, {"configuration": {"seed": 17}, "data": {"retained_rows": 1247}})
        self.assertEqual(phase["configuration"]["max_steps"], 300)
        self.assertEqual(phase["configuration"]["max_training_seconds"], 7200)
        self.assertTrue(phase["configuration"]["reuse_loaded_model"])
        self.assertTrue(phase["configuration"]["optimizer_reset"])
        self.assertTrue(phase["configuration"]["sampler_reset"])
        self.assertEqual(phase["initial_adapter"]["weights_sha256"], "a" * 64)

    def test_kaggle_route_uses_single_process_continuation(self):
        source = (CASE / "run_kaggle_route.py").read_text(encoding="utf-8")
        self.assertIn('"--continuation-out"', source)
        self.assertIn('"--continuation-steps"', source)
        self.assertIn('"--continuation-training-seconds"', source)
        self.assertIn("train_with_session", source)
        self.assertIn("HfStrategyRouter.from_preloaded", source)
        self.assertIn('"base_model_reloaded": False', source)
        self.assertIn('"--max-generation-seconds"', source)
        self.assertIn('"--max-inference-seconds"', source)
        self.assertIn('"4608"', source)
        self.assertIn('"--prompt-encoding"', source)
        self.assertIn('"compact-grid-v1"', source)
        self.assertNotIn('"--initial-adapter"', source)

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

    def test_kaggle_runtime_ignores_rewritten_control_plane_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "private_data/train.jsonl"
            payload.parent.mkdir()
            payload.write_bytes(b"sealed payload")
            metadata = root / "dataset-metadata.json"
            metadata.write_text('{"id":"owner/source"}\n', encoding="utf-8")
            manifest = {
                "format_version": "hfr.arcagi.kaggle_private_bundle.v1",
                "publication_allowed": False,
                "files": {
                    "private_data/train.jsonl": {
                        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                        "size_bytes": payload.stat().st_size,
                    }
                },
            }
            (root / "bundle_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            metadata.write_text('{"id":"provider/normalized"}\n', encoding="utf-8")
            self.assertEqual(kaggle_route.verify_bundle(root), manifest)

            payload.write_bytes(b"tampered payload")
            with self.assertRaisesRegex(ValueError, "private bundle integrity failure"):
                kaggle_route.verify_bundle(root)

    def test_kaggle_bundle_seals_payload_but_not_provider_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "payload.bin").write_bytes(b"sealed")
            (root / "dataset-metadata.json").write_text("{}\n", encoding="utf-8")
            (root / "bundle_manifest.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                set(kaggle_bundle.sealed_payload_inventory(root)),
                {"payload.bin"},
            )

    def test_kaggle_notebook_installs_only_pinned_offline_wheels(self):
        source = "".join(kaggle_bundle.notebook_payload()["cells"][0]["source"])
        compile(source, "arcagi_kaggle_route.ipynb", "exec")
        self.assertIn("--no-index", source)
        self.assertIn("--no-deps", source)
        self.assertIn(kaggle_bundle.PEFT_IMPORT_UTILS_SHA256, source)
        self.assertIn("disable_unused_torchao_dispatcher", source)
        self.assertIn("pinned PEFT import-utils source hash mismatch", source)
        self.assertIn("deterministic_t4_compact_context_suffix_logits", source)
        self.assertIn("sealed Kaggle route source hash mismatch", source)
        self.assertIn("runner_source.count(compact_context_training) != 1", source)
        self.assertIn("hfr_run_kaggle_route_gpu_compact_4608.py", source)
        self.assertIn("compact-context runner copy hash mismatch", source)
        self.assertIn("'max_seq_length': 4608", source)
        self.assertIn("'minimum_retained_fraction': 1.0", source)
        self.assertIn("'prompt_encoding': 'compact-grid-v1'", source)
        self.assertNotIn("bounded_context_training", source)
        self.assertNotIn("deterministic_cpu_training_gpu_inference", source)
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

    def test_kaggle_runtime_overlay_contains_only_reviewed_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runtime-overlay"
            status = kaggle_runtime_overlay.build(
                SimpleNamespace(
                    repo_root=ROOT,
                    dataset_id="owner/hfr-arc-runtime-v1",
                    out=out,
                    adapter_root=None,
                )
            )
            self.assertEqual(status, 0)
            manifest = json.loads(
                (out / "runtime_patch_manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["publication_allowed"])
            self.assertFalse(manifest["contains_raw_traces"])
            self.assertFalse(manifest["contains_arc_tasks"])
            self.assertEqual(
                set(manifest["files"]),
                {f"case/{name}" for name in kaggle_runtime_overlay.RUNTIME_FILES},
            )
            self.assertEqual(
                {path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()},
                {
                    "dataset-metadata.json",
                    "runtime_patch_manifest.json",
                    *manifest["files"],
                },
            )

    def test_kaggle_runtime_overlay_binds_frozen_adapter_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            config = adapter / "adapter_config.json"
            weights = adapter / "adapter_model.safetensors"
            config.write_bytes(b"config")
            weights.write_bytes(b"weights")
            receipt = {
                "status": "succeeded",
                "target_steps": 900,
                "completed_steps": 900,
                "outputs": {
                    "adapter_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "adapter_model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                },
            }
            (adapter / "training_receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            out = root / "runtime-overlay"
            status = kaggle_runtime_overlay.build(
                SimpleNamespace(
                    repo_root=ROOT,
                    dataset_id="owner/hfr-arc-runtime-v2",
                    out=out,
                    adapter_root=adapter,
                    adapter_source_kernel="owner/kernel",
                    adapter_source_version=12,
                )
            )
            self.assertEqual(status, 0)
            manifest = json.loads(
                (out / "runtime_patch_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["contains_model_artifacts"])
            self.assertEqual(manifest["adapter"]["source_version"], 12)
            self.assertIn(
                "adapter/checkpoint-0900/adapter_model.safetensors",
                manifest["files"],
            )

    def test_kaggle_comparison_notebook_is_private_offline_and_hash_pinned(self):
        source = "".join(
            kaggle_runtime_overlay.comparison_notebook_payload("a" * 64)["cells"][0][
                "source"
            ]
        )
        compile(source, "arcagi_kaggle_comparison.ipynb", "exec")
        self.assertIn("private comparison overlay manifest hash mismatch", source)
        self.assertIn("base_vs_hfr_lora_router_only_identical_executor", source)
        self.assertIn("router-only", source)
        self.assertIn("--no-index", source)
        self.assertIn("str(runtime_case), str(vendor)", source)
        self.assertNotIn("https://", source)
        metadata = kaggle_runtime_overlay.comparison_kernel_metadata_payload(
            bundle_dataset_id="owner/private-bundle",
            overlay_dataset_id="owner/private-overlay",
            kernel_id="owner/private-comparison",
            competition="arc-prize-2026-arc-agi-2",
        )
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(metadata["enable_internet"], "false")
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(
            metadata["dataset_sources"],
            ["owner/private-bundle", "owner/private-overlay"],
        )

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

    def test_compact_arc_prompt_is_lossless_and_versioned(self):
        task = {
            "train": [
                {
                    "input": [[0, 1], [2, 3]],
                    "output": [[3, 2], [1, 0]],
                }
            ],
            "test": [{"input": [[4, 5], [6, 7]]}],
        }
        native = pipeline.prompt_for(task)
        compact = pipeline.compact_prompt_from_native(native)
        self.assertEqual(compact, pipeline.compact_prompt_for(task))
        self.assertIn("ARC_COMPACT_V1", compact)
        self.assertIn("01/23>32/10", compact)
        self.assertIn("test_input=45/67", compact)

    def test_peft_compact_encoding_preserves_recorded_row(self):
        task = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 3]]}],
        }
        row = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": pipeline.prompt_for(task)},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "tool"}}]},
            ],
            "tools": [{"type": "function"}],
        }
        encoded = peft_trainer.training_row_for_prompt_encoding(row, "compact-grid-v1")
        self.assertEqual(row["messages"][1]["content"], pipeline.prompt_for(task))
        self.assertEqual(encoded["messages"][1]["content"], pipeline.compact_prompt_for(task))


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

    def test_kaggle_public_evidence_is_sanitized_and_hash_bound(self):
        metrics = json.loads(
            (KAGGLE_PUBLIC_RESULTS / "metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["visible_results"]["base"]["exact_grid_count"], 0)
        self.assertEqual(
            metrics["visible_results"]["lora_checkpoint_0900"]["exact_grid_count"],
            17,
        )
        self.assertEqual(metrics["sealed_artifacts"]["test_input_count"], 259)
        self.assertTrue(metrics["sealed_artifacts"]["base_gate_passed"])
        self.assertTrue(metrics["sealed_artifacts"]["lora_gate_passed"])
        self.assertEqual(
            metrics["kaggle"]["lora_submission_status"], "complete_unscored"
        )
        self.assertIsNone(metrics["kaggle"]["lora_public_score"])
        self.assertIsNone(metrics["kaggle"]["lora_private_score"])
        self.assertFalse(metrics["kaggle"]["official_improvement_claim_allowed"])
        forbidden_keys = {
            "episode_id",
            "task_family",
            "raw_output",
            "selected_strategy",
        }

        def walk(value):
            if isinstance(value, dict):
                self.assertTrue(forbidden_keys.isdisjoint(value))
                for nested in value.values():
                    walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    walk(nested)
            elif isinstance(value, str):
                self.assertNotIn("/Users/", value)

        walk(metrics)

    def test_kaggle_public_checksum_manifest_verifies(self):
        checksums = {}
        for line in (KAGGLE_PUBLIC_RESULTS / "SHA256SUMS").read_text(
            encoding="utf-8"
        ).splitlines():
            digest, filename = line.split("  ", 1)
            checksums[filename] = digest

        expected_files = {"metrics.json", "metrics_summary.csv"}
        self.assertEqual(set(checksums), expected_files)
        for filename, expected_digest in checksums.items():
            actual_digest = hashlib.sha256(
                (KAGGLE_PUBLIC_RESULTS / filename).read_bytes()
            ).hexdigest()
            self.assertEqual(actual_digest, expected_digest)


if __name__ == "__main__":
    unittest.main()
