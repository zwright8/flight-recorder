# Architecture

Flight Recorder supports Python 3.11 and later. The core package uses only the
standard library: `argparse` provides the CLI and `unittest` provides the test
suite. `flightrecorder` and `flight-recorder` call `flightrecorder.cli:main`;
`python -m flightrecorder` uses that same entry point.

## Boundaries

| Boundary | Implementation | Responsibility |
| --- | --- | --- |
| File hashing | `flightrecorder.hashing` | `sha256_file()` is the shared leaf helper. It reads a file in 1 MiB binary chunks and returns its lowercase SHA-256 digest. Existing private helper names remain local aliases. |
| Artifact validation | `flightrecorder.validation`, `flightrecorder._validation` | The public facade preserves validator imports, constants, `ValidationTarget`, and the explicit keyword-only `validate_artifacts()` signature. Private modules split shared primitives, dispatch, and artifact-domain checks. |
| Schema and source contracts | `flightrecorder.schema_registry`, `flightrecorder.source_contract` | The schema registry checks bundled JSON schema shapes independently. Source-contract inspection defers semantic-validator lookup through the validation facade, avoiding an eager import cycle. |
| CLI | `flightrecorder.cli`, `flightrecorder._cli` | The facade owns the console entry point, exception-to-exit behavior, parser access, and script compatibility aliases. Private modules hold commands, parser registrars, and shared run, replay, and suite mechanics. |

`_validation/constants.py` holds shared contract values, and
`_validation/primitives.py` owns common readers, safe-path checks, and target
construction. `dispatch.py` keeps `validate_artifacts()`' ordered aggregation.
Domain validators live in `runs.py`, `exports.py`, `review.py`,
`governance.py`, `models.py`, `training_plan_runtime.py`,
`training_flow_result.py`, `training_loop.py`, `cloud.py`,
`trainer_archive.py`, and `evaluation_serving.py`. The explicit parameter list
and target dispatch order are part of the validation contract.

`_cli/parser.py` composes the parser from ordered registrars in `artifacts.py`,
`basic.py`, `evals.py`, `evidence.py`, `exports.py`, `gates.py`,
`generation.py`, `governance.py`, `loop.py`, `models.py`, `run_commands.py`,
and `suite.py`. `constants.py`, `shared.py`, `run_core.py`, `replay_core.py`,
and `suite_core.py` hold lower-level mechanics. The public facade retains
`main`, `_parser`, `ReplayError`, all `cmd_*` handlers, and script compatibility
aliases such as `_run_scenario_artifacts`, `_run_suite_summary`, and
`_write_json`. Registrar order is intentional because command order and help
output are observable.

## Extending the system

To add a command, implement the handler in the appropriate `_cli` domain,
register its arguments through that domain's registrar, and add the registrar
at the required position in `_cli/parser.py`. Keep the facade import when a
repository script or supported caller imports the handler or helper. Test it
through `python -m flightrecorder` or `flightrecorder.cli.main()` and inspect
the command's output artifact and exit status.

To add a validator, use `_validation/primitives.py` for common reading and
pathing, place semantic checks in the relevant domain module, and re-export a
supported validator from `flightrecorder.validation`. Add its path category to
the explicit ordered dispatch only where its output must appear. Preserve the
deferred facade lookup when adding a source-contract role; do not import
`source_contract` from the schema registry.

Put regression coverage under `tests/`, beside the contract it exercises. Test
public imports and command behavior as well as a representative serialized
artifact and failure case. Use temporary directories so tests do not overwrite
user artifacts.

## Compatibility rules

This organization is structural. It must preserve command names and parser
order; error types and asserted messages; strict-mode behavior; validation
target and error order; redaction and safe-path checks; SHA-256 values; schema
contracts; and atomic-write behavior. New code must not turn a deferred
validation lookup into an import cycle or replace an atomic write with a direct
overwrite.
