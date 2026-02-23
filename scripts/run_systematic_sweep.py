#!/usr/bin/env python
"""Execute a systematic sweep manifest with resume and retry semantics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.sweep import append_jsonl, load_jsonl, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        required=True,
        help="Sweep root directory containing manifest and generated configs.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path override. Defaults to <sweep-root>/manifest.jsonl.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help=(
            "Optional run registry JSONL path override. "
            "Defaults to <sweep-root>/run_registry.jsonl."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry count after first failed attempt for each config.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Optional cap on number of configs to execute from manifest order.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing remaining configs when a config fails all retries.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run configs even if they already have success status in registry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned executions without running commands.",
    )
    return parser.parse_args()


def _manifest_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("stage", "")), str(row.get("config_id", "")))


def _latest_status_by_config(registry_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in registry_rows:
        config_id = str(row.get("config_id", ""))
        if not config_id:
            continue
        latest[config_id] = row
    return latest


def _normalize_commands(row: dict[str, Any]) -> list[dict[str, Any]]:
    commands = row.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError(f"commands must be a list for config_id={row.get('config_id')}")
    normalized: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        if isinstance(command, dict) and isinstance(command.get("args"), list):
            normalized.append(
                {
                    "name": str(command.get("name", f"command_{index}")),
                    "args": [str(token) for token in command["args"]],
                }
            )
            continue
        if isinstance(command, list):
            normalized.append(
                {
                    "name": f"command_{index}",
                    "args": [str(token) for token in command],
                }
            )
            continue
        raise ValueError(
            f"Unsupported command format in manifest for config_id={row.get('config_id')}"
        )
    return normalized


def _resolve_python_executable(args: list[str]) -> list[str]:
    if not args:
        return args
    executable = str(args[0]).strip()
    if executable in {"python", "python3"}:
        return [sys.executable, *args[1:]]
    return args


def _write_step_log(
    log_path: Path, *, args: list[str], return_code: int, stdout: str, stderr: str
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"command: {' '.join(args)}",
        f"return_code: {return_code}",
        "",
        "=== STDOUT ===",
        stdout,
        "",
        "=== STDERR ===",
        stderr,
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")


def _run_step(
    *,
    args: list[str],
    cwd: Path,
    log_path: Path,
) -> tuple[int, str, str]:
    resolved_args = _resolve_python_executable(args)
    completed = subprocess.run(
        resolved_args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    _write_step_log(
        log_path,
        args=resolved_args,
        return_code=int(completed.returncode),
        stdout=stdout,
        stderr=stderr,
    )
    return int(completed.returncode), stdout, stderr


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    sweep_root = Path(args.sweep_root)
    if not sweep_root.is_absolute():
        sweep_root = repo_root / sweep_root

    manifest_path = Path(args.manifest) if args.manifest else sweep_root / "manifest.jsonl"
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    registry_path = Path(args.registry) if args.registry else sweep_root / "run_registry.jsonl"
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path

    manifest_rows = sorted(load_jsonl(manifest_path), key=_manifest_sort_key)
    if args.max_configs is not None:
        manifest_rows = manifest_rows[: int(args.max_configs)]

    registry_rows = load_jsonl(registry_path)
    latest_status = _latest_status_by_config(registry_rows)

    planned: list[dict[str, Any]] = []
    for row in manifest_rows:
        config_id = str(row.get("config_id", ""))
        if not config_id:
            continue
        if not args.force:
            status_payload = latest_status.get(config_id, {})
            if str(status_payload.get("status", "")) == "success":
                continue
        planned.append(row)

    if args.dry_run:
        preview = {
            "manifest_path": str(manifest_path),
            "registry_path": str(registry_path),
            "planned_configs": [row["config_id"] for row in planned],
            "count": len(planned),
        }
        print(json.dumps(preview, indent=2, sort_keys=True))
        return

    success_count = 0
    failed_count = 0

    for row in planned:
        config_id = str(row["config_id"])
        stage = str(row.get("stage", "unknown"))
        run_root = Path(str(row.get("run_root", "")))
        if not run_root.is_absolute():
            run_root = repo_root / run_root

        commands = _normalize_commands(row)

        config_success = False
        for attempt in range(1, int(args.retries) + 2):
            started_at = time.time()
            append_jsonl(
                registry_path,
                {
                    "config_id": config_id,
                    "stage": stage,
                    "status": "running",
                    "attempt": attempt,
                    "start_time": utc_now_iso(),
                    "end_time": None,
                    "duration_sec": None,
                    "error_type": None,
                    "error_message": None,
                    "artifacts": {
                        "config_path": row.get("config_path"),
                        "run_root": str(run_root),
                    },
                },
            )

            failure_type: str | None = None
            failure_message: str | None = None
            failed_step: str | None = None
            for step in commands:
                step_name = str(step["name"])
                step_args = [str(token) for token in step["args"]]
                step_log_path = (
                    sweep_root / "logs" / config_id / f"attempt_{attempt}_{step_name}.log"
                )
                return_code, stdout, stderr = _run_step(
                    args=step_args,
                    cwd=repo_root,
                    log_path=step_log_path,
                )
                if return_code != 0:
                    failure_type = "command_failed"
                    tail = (stderr or stdout)[-1200:]
                    failure_message = (
                        f"step={step_name} return_code={return_code} tail={tail.strip()}"
                    )
                    failed_step = step_name
                    break

            ended_at = time.time()
            duration_sec = round(ended_at - started_at, 3)

            if failure_type is None:
                append_jsonl(
                    registry_path,
                    {
                        "config_id": config_id,
                        "stage": stage,
                        "status": "success",
                        "attempt": attempt,
                        "start_time": None,
                        "end_time": utc_now_iso(),
                        "duration_sec": duration_sec,
                        "error_type": None,
                        "error_message": None,
                        "artifacts": {
                            "config_path": row.get("config_path"),
                            "run_root": str(run_root),
                            "aggregate_json": str(run_root / "aggregate_lopo_metrics.json"),
                            "aggregate_md": str(run_root / "aggregate_lopo_metrics.md"),
                        },
                    },
                )
                config_success = True
                success_count += 1
                break

            append_jsonl(
                registry_path,
                {
                    "config_id": config_id,
                    "stage": stage,
                    "status": "failed",
                    "attempt": attempt,
                    "start_time": None,
                    "end_time": utc_now_iso(),
                    "duration_sec": duration_sec,
                    "error_type": failure_type,
                    "error_message": failure_message,
                    "artifacts": {
                        "config_path": row.get("config_path"),
                        "run_root": str(run_root),
                        "failed_step": failed_step,
                    },
                },
            )

        if not config_success:
            failed_count += 1
            if not args.continue_on_error:
                raise RuntimeError(
                    f"Config {config_id} failed after {int(args.retries) + 1} attempts"
                )

    summary = {
        "manifest_path": str(manifest_path),
        "registry_path": str(registry_path),
        "planned_configs": len(planned),
        "succeeded": success_count,
        "failed": failed_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
