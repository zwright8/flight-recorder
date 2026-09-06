import inspect
import importlib
import tempfile
import unittest
from pathlib import Path

from flightrecorder import source_contract, validation


class ValidationStructureTests(unittest.TestCase):
    def test_facade_reexports_every_extracted_callable_by_identity(self):
        module_names = (
            "cloud",
            "dispatch",
            "evaluation_serving",
            "exports",
            "governance",
            "models",
            "primitives",
            "review",
            "runs",
            "trainer_archive",
            "training_flow_result",
            "training_loop",
            "training_plan_runtime",
        )
        missing = []
        mismatched = []
        for module_name in module_names:
            module = importlib.import_module(
                f"flightrecorder._validation.{module_name}"
            )
            for name, value in vars(module).items():
                if not (inspect.isfunction(value) or inspect.isclass(value)):
                    continue
                if getattr(value, "__module__", None) != module.__name__:
                    continue
                facade_value = getattr(validation, name, None)
                if facade_value is None:
                    missing.append(name)
                elif facade_value is not value:
                    mismatched.append(name)

        self.assertEqual(missing, [])
        self.assertEqual(mismatched, [])

    def test_validate_artifacts_keyword_contract_is_stable(self):
        signature = inspect.signature(validation.validate_artifacts)

        self.assertTrue(signature.parameters)
        self.assertTrue(
            all(
                parameter.kind is inspect.Parameter.KEYWORD_ONLY
                for parameter in signature.parameters.values()
            )
        )
        self.assertEqual(next(reversed(signature.parameters)), "strict")
        self.assertEqual(signature.parameters["strict"].default, False)

    def test_validate_artifacts_rejects_empty_configuration(self):
        result = validation.validate_artifacts()

        self.assertEqual(
            result,
            {
                "schema_version": "hfr.validation.v1",
                "passed": False,
                "strict": False,
                "target_count": 1,
                "error_count": 1,
                "warning_count": 0,
                "targets": [
                    {
                        "type": "configuration",
                        "path": ".",
                        "passed": False,
                        "errors": ["No validation targets configured."],
                        "warnings": [],
                        "details": {},
                    }
                ],
            },
        )

    def test_validate_artifacts_preserves_expansion_and_scalar_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = validation.validate_artifacts(
                run_dirs=[root / "run-b", root / "run-a"],
                runs_dir=root / "runs",
                training_export_dir=root / "training",
                compare_export_dir=root / "compare",
            )

        self.assertEqual(
            [target["type"] for target in result["targets"]],
            ["run", "run", "runs", "training_export", "compare_export"],
        )
        self.assertEqual(
            [Path(target["path"]).name for target in result["targets"]],
            ["run-b", "run-a", "runs", "training", "compare"],
        )

    def test_all_deferred_source_contract_validators_are_exported(self):
        missing = [
            name
            for name in source_contract._SEMANTIC_VALIDATOR_NAMES.values()
            if not callable(getattr(validation, name, None))
        ]

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
