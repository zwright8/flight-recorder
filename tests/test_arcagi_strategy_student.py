import importlib.util
import json
import sys
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
else:
    pipeline = None
    evaluation = None
    ensemble = None


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


class ArcAgiPublicEvidenceTests(unittest.TestCase):
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
