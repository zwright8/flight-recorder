from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import flightrecorder.cli as cli
from flightrecorder._cli.artifacts import VALIDATE_APPEND_OPTIONS
from flightrecorder.validation import validate_artifacts


EXPECTED_PARSER_CONTRACT_SHA256 = "672f4c6c9d23465fafcf6ab1e525855f1313d709a03569e79eeebd17ca17ca69"
EXPECTED_COMMAND_ORDER = (
    "normalize", "score", "report", "digest", "capture-state", "verify-state",
    "state-validators", "diff-state", "run", "replay", "replay-bundle", "run-suite",
    "goal3-handoff", "index", "audit", "compare", "compare-suite", "trend-suite",
    "check-scenarios", "scenario-quality", "validate", "model-scout", "model-candidate",
    "model-registry", "training-plan", "heldout-manifest", "external-eval-plan",
    "external-eval-receipt", "external-eval-result", "data-governance", "intervention-route",
    "runtime-router", "review-semantics", "agentic-loop", "next-iteration-schedule",
    "agentic-training-flow", "cloud-training", "agentic-rollout-plan",
    "agentic-rollout-receipt", "rejection-sampling-gate", "dataset-curation-receipt",
    "model-grader", "eval-summary", "schemas", "evidence-coverage", "trace-observability",
    "repair-queue", "evidence-bundle", "improvement-plan", "improvement-ledger",
    "gate-improvement-ledger", "action-ledger", "promotion-cards", "promotion-alias-apply",
    "promotion-rollback-receipt", "promotion-release-record", "promotion-decision",
    "promotion-ledger", "promotion-archive", "gate-promotion-ledger", "gate-action-ledger",
    "gate-decision", "draft-scenario", "gate-suite", "gate-export", "gate-reviewed",
    "gate-compare-export", "trainer-preflight", "trainer-launch-check", "trainer-archive",
    "trainer-archive-check", "trainer-consumer-plan", "export-rl", "export-compare-rl",
    "export-review", "apply-review", "review-calibration", "observer-template",
)
EXPECTED_COMPATIBILITY_EXPORTS = (
    "_failed_rule_ids",
    "_lineage_input_hash",
    "_is_owned_run_directory",
    "_parser",
    "_run_scenario_artifacts",
    "_run_suite_summary",
    "_safe_run_id",
    "_sha256_file",
    "_write_json",
    "cmd_replay",
    "main",
    "ReplayError",
)


def _contract_value(value):
    if callable(value):
        return {"callable": getattr(value, "__name__", type(value).__name__)}
    if isinstance(value, (list, tuple)):
        return [_contract_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _parser_contract(parser: argparse.ArgumentParser):
    actions = []
    children = {}
    for action in parser._actions:
        actions.append(
            {
                "action": type(action).__name__,
                "dest": action.dest,
                "options": list(action.option_strings),
                "required": action.required,
                "nargs": _contract_value(action.nargs),
                "const": _contract_value(action.const),
                "default": _contract_value(action.default),
                "type": _contract_value(action.type),
                "choices": _contract_value(list(action.choices) if action.choices is not None else None),
                "metavar": _contract_value(action.metavar),
                "help": action.help,
            }
        )
        if isinstance(action, argparse._SubParsersAction):
            children = {name: _parser_contract(child) for name, child in action.choices.items()}
    return {
        "prog": parser.prog,
        "description": parser.description,
        "actions": actions,
        "children": children,
    }


class CliStructureTests(unittest.TestCase):
    def test_cli_facade_keeps_compatibility_exports(self) -> None:
        for name in EXPECTED_COMPATIBILITY_EXPORTS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(cli, name))
        self.assertEqual(cli.main.__module__, "flightrecorder.cli")
        self.assertEqual(cli._parser.__module__, "flightrecorder.cli")
        self.assertTrue(cli.cmd_replay.__module__.startswith("flightrecorder._cli."))

    def test_private_cli_modules_do_not_depend_on_facade(self) -> None:
        package_dir = Path(cli.__file__).with_name("_cli")
        for module_path in package_dir.glob("*.py"):
            source = module_path.read_text(encoding="utf-8")
            with self.subTest(module=module_path.name):
                self.assertNotIn("flightrecorder.cli", source)
                self.assertNotIn("from ..cli", source)
                self.assertNotIn("Path(__file__).resolve().parents[1]", source)

    def test_validate_append_table_matches_parser_and_validator_signature(self) -> None:
        self.assertEqual(len(VALIDATE_APPEND_OPTIONS), 86)
        self.assertEqual(len({dest for _, dest, _, _ in VALIDATE_APPEND_OPTIONS}), 86)
        self.assertEqual(len({keyword for _, _, keyword, _ in VALIDATE_APPEND_OPTIONS}), 86)
        validator_parameters = inspect.signature(validate_artifacts).parameters
        for _, _, keyword, _ in VALIDATE_APPEND_OPTIONS:
            self.assertIn(keyword, validator_parameters)

        parser = cli._parser()
        root = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        validate_parser = root.choices["validate"]
        append_options = tuple(
            action.option_strings[0]
            for action in validate_parser._actions
            if isinstance(action, argparse._AppendAction)
        )
        self.assertEqual(append_options, tuple(flag for flag, _, _, _ in VALIDATE_APPEND_OPTIONS))

    def test_parser_contract_matches_characterized_baseline(self) -> None:
        parser = cli._parser()
        payload = json.dumps(_parser_contract(parser), sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), EXPECTED_PARSER_CONTRACT_SHA256)

    def test_top_level_command_order_matches_characterized_baseline(self) -> None:
        parser = cli._parser()
        subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        self.assertEqual(tuple(subparsers.choices), EXPECTED_COMMAND_ORDER)

    def test_append_defaults_keep_argparse_mutable_default_behavior(self) -> None:
        parser = cli._parser()
        first = parser.parse_args(["normalize", "--trace", "trace.json", "--out", "out.json"])
        second = parser.parse_args(["normalize", "--trace", "trace.json", "--out", "out.json"])
        third = cli._parser().parse_args(["normalize", "--trace", "trace.json", "--out", "out.json"])
        self.assertIs(first.secret_pattern, second.secret_pattern)
        self.assertIsNot(first.secret_pattern, third.secret_pattern)

    def test_main_preserves_replay_error_and_interrupt_exit_contracts(self) -> None:
        class StubParser:
            def __init__(self, error: BaseException) -> None:
                self.error = error

            def parse_args(self, argv):
                return argparse.Namespace(func=lambda args: (_ for _ in ()).throw(self.error))

            def exit(self, status=0, message=None):
                if message:
                    print(message, end="", file=__import__("sys").stderr)
                raise SystemExit(status)

        for error, expected_status, expected_message in (
            (cli.ReplayError("bad replay"), 2, "flightrecorder: error: bad replay\n"),
            (KeyboardInterrupt(), 130, "flightrecorder: interrupted\n"),
        ):
            with self.subTest(error=type(error).__name__):
                stderr = io.StringIO()
                with patch.object(cli, "_parser", return_value=StubParser(error)), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        cli.main([])
                self.assertEqual(raised.exception.code, expected_status)
                self.assertEqual(stderr.getvalue(), expected_message)

    def test_replay_loader_preserves_caller_specific_failure_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lineage = Path(tmp) / "artifact_lineage.json"
            lineage.write_text(
                json.dumps({"replay": {"self_contained": False, "argv": "invalid"}}),
                encoding="utf-8",
            )
            cases = (
                (
                    ["replay", "--lineage", str(lineage), "--out", str(Path(tmp) / "run")],
                    "replay contract is not self-contained",
                ),
                (
                    ["replay-bundle", "--lineage", str(lineage), "--out", str(Path(tmp) / "bundle")],
                    "artifact_lineage.replay.argv must be a list of strings",
                ),
            )
            for argv, expected in cases:
                with self.subTest(command=argv[0]):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                        cli.main(argv)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(expected, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
