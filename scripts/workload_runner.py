#!/usr/bin/env python3
"""Internal workload-runner entrypoint; not a general command interface."""

import argparse
import json
import os
import sys
import time

from runner_supervisor import RUNNER_EVENT_PREFIX, SUPPORTED_RUNNER_STAGES


def emit(kind: str, **details) -> None:
    payload = {
        "ownership_token": os.environ.get("LOCAL_AI_BENCH_RUNNER_TOKEN"),
        "kind": kind, "timestamp": time.time(), **details,
    }
    sys.stdout.write(f"{RUNNER_EVENT_PREFIX}{json.dumps(payload, separators=(',', ':'))}\n")
    sys.stdout.flush()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Internal Local AI Bench workload runner")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(SUPPORTED_RUNNER_STAGES))
    parser.add_argument("--event-store", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.environ.get("LOCAL_AI_BENCH_RUNNER_TOKEN"):
        sys.stderr.write("Runner ownership token is required.\n")
        return 2
    emit("terminal", status="not_activated", job_id=args.job_id, stage=args.stage)
    return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
