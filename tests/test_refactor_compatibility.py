from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from flightrecorder import cli, training, validation

ROOT = Path(__file__).resolve().parents[1]
FIXED_NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
CLI_CONTRACT_SHA256 = "947204336772c0452d36a04dc479fdc75aa466362573becb3c9225f01595a26c"
VALIDATION_CONTRACT_SHA256 = "987054a33cf7245712e72d78ec856c14e9f7765693f17e82d4f0b5ca53537f7d"
VALIDATION_CONSTANTS_SHA256 = "a3f40d5e94c979d46e23b23674819c74233123629710f40b13b7040afbef9b18"
REQUIRED_PRIVATE_CLI = {
    "_failed_rule_ids",
    "_is_owned_run_directory",
    "_lineage_input_hash",
    "_parser",
    "_run_scenario_artifacts",
    "_run_suite_summary",
    "_safe_run_id",
    "_write_json",
}
EXPECTED_OUTPUTS = (
    "PASS prompt_injection_good score=100 report=runs/good/report.html\n",
    "PASS replay prompt_injection_good score=100 out=runs/replay\n",
    "FAIL prompt_injection_bad score=0 report=runs/bad/report.html\n",
    "wrote RL export episodes=3 rewards=3 step_rewards=11 preferences=2 failure_modes=4 "
    "sft=2 dpo=2 reward_model=3 quality_flags=3 out=training_export\n",
    "wrote validation.json\n",
)
EXPECTED_ARTIFACT_SHA256 = {
    "runs/bad/normalized_trace.json": "723c77584e15d80e925b15aebebe04a3ffb0a6be8940e46b80948bb7de7e622c",
    "runs/bad/scorecard.json": "7e816d3b9d6722c3435923b977767eb9a22b035e2234c180d65702917f4914b0",
    "runs/good/artifact_lineage.json": "855a62d7a138449db15ce4930ebaa4625e9755df750d723f24f9f73488d61b23",
    "runs/good/normalized_trace.json": "6dc71608ef97977c0861dd83a910ad42ba5bc03ab975578e32cbc6fe7e0c2c71",
    "runs/good/run_digest.json": "4ad2410e1c6c1dc1f9db636214a25412e70c9ce1d144176b467546d3452e6798",
    "runs/good/scorecard.json": "1564c0ea4d601eba38a2712dfff6a8eae3eb4bee075e9ba31b07374c71a66d9d",
    "runs/replay/artifact_lineage.json": "d91d1de95e6355e27cb739d8ec5598493bfbbc127a86797fd698015e009958ec",
    "runs/replay/normalized_trace.json": "6dc71608ef97977c0861dd83a910ad42ba5bc03ab975578e32cbc6fe7e0c2c71",
    "runs/replay/scorecard.json": "1564c0ea4d601eba38a2712dfff6a8eae3eb4bee075e9ba31b07374c71a66d9d",
    "training_export/dataset_metrics.json": "ecf227e2d438cd5272e0bba66e2a920ba61c6a0b7b51e1d3ee3db43502daa16d",
    "training_export/dpo.jsonl": "fea5541854a360fabe51c5d052ca96538e2c23321e538ae1b915a68e63b5b58b",
    "training_export/episodes.jsonl": "22391b720c019d53edb8e3222b5f15e9ea84ded0a2dfc65505caa914ad9cfae6",
    "training_export/manifest.json": "2b91a969db5438ad97723781e474dada23a1dee65772912dc5abfc906728d02e",
    "validation.json": "4f4da86dbfe00ebda9ec84bd94f99b45f6e0450ea3998ec06cbfaa4b42ed454f",
}


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is not None else FIXED_NOW.replace(tzinfo=None)


def _invoke(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        returncode = cli.main(argv)
    return returncode, stdout.getvalue(), stderr.getvalue()


def _callable_contract(module, names: set[str]) -> str:
    contract = {}
    for name in sorted(names):
        value = getattr(module, name, None)
        if value is None:
            contract[name] = None
            continue
        try:
            signature = str(inspect.signature(value))
        except (TypeError, ValueError):
            signature = None
        contract[name] = ["class" if inspect.isclass(value) else "function", signature]
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _constant_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_constant_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_constant_value(item) for item in value)
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _constant_value(item) for key, item in sorted(value.items())}
    raise TypeError


class RefactorCompatibilityTests(unittest.TestCase):
    def test_facades_match_characterized_callable_and_constant_contracts(self) -> None:
        # Given: the callable and constant names exposed by the pinned baseline facades.
        cli_names = {
            name for name in dir(cli)
            if name in {"main", "ReplayError"} or name.startswith("cmd_")
        } | REQUIRED_PRIVATE_CLI
        validation_names = {
            name for name in dir(validation)
            if name == "ValidationTarget" or name.startswith("validate_")
        }
        validation_constants = {}
        for name in dir(validation):
            if not name.startswith("_") and name.isupper():
                try:
                    validation_constants[name] = _constant_value(getattr(validation, name))
                except TypeError:
                    pass

        # When: the current public facades are reduced to machine-consumed signatures and values.
        actual = {
            "cli": _callable_contract(cli, cli_names),
            "validation": _callable_contract(validation, validation_names),
            "validation_constants": hashlib.sha256(
                json.dumps(validation_constants, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }

        # Then: every facade contract matches the pinned baseline.
        self.assertEqual(
            actual,
            {
                "cli": CLI_CONTRACT_SHA256,
                "validation": VALIDATION_CONTRACT_SHA256,
                "validation_constants": VALIDATION_CONSTANTS_SHA256,
            },
        )

    def test_run_replay_export_and_validation_match_characterized_artifacts(self) -> None:
        # Given: copied repository fixtures, relative paths, and a fixed export timestamp.
        with tempfile.TemporaryDirectory(prefix="hfr-refactor-compatibility-") as tmp:
            root = Path(tmp)
            (root / "scenarios").mkdir()
            (root / "fixtures").mkdir()
            for name in ("prompt_injection_good.json", "prompt_injection_bad.json"):
                shutil.copy2(ROOT / "scenarios" / name, root / "scenarios" / name)
            for name in ("prompt_injection_good.trajectory.jsonl", "prompt_injection_bad.trajectory.jsonl"):
                shutil.copy2(ROOT / "fixtures" / name, root / "fixtures" / name)

            # When: the representative CLI workflow executes through its real command surface.
            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                with patch.object(training, "datetime", FixedDateTime):
                    results = (
                        _invoke(["run", "--scenario", "scenarios/prompt_injection_good.json", "--out", "runs/good"]),
                        _invoke(["replay", "--lineage", "runs/good/artifact_lineage.json", "--out", "runs/replay", "--base-dir", "."]),
                        _invoke(["run", "--scenario", "scenarios/prompt_injection_bad.json", "--out", "runs/bad"]),
                        _invoke(["export-rl", "--runs", "runs", "--out", "training_export", "--metadata", "characterization=baseline"]),
                        _invoke(["validate", "--runs", "runs", "--training-export", "training_export", "--strict", "--out", "validation.json"]),
                    )
            finally:
                os.chdir(old_cwd)

            artifact_hashes = {
                path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                for path in EXPECTED_ARTIFACT_SHA256
            }

            # Then: exits, output streams, serialization, fingerprints, and validation are unchanged.
            self.assertEqual(tuple(code for code, _, _ in results), (0, 0, 0, 0, 0))
            self.assertEqual(tuple(stdout for _, stdout, _ in results), EXPECTED_OUTPUTS)
            self.assertEqual(tuple(stderr for _, _, stderr in results), ("", "", "", "", ""))
            self.assertEqual(artifact_hashes, EXPECTED_ARTIFACT_SHA256)


if __name__ == "__main__":
    unittest.main()
