#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    base = Path(__file__).resolve().parent
    shared_runner = base / "two_tenant_cost_run.py"
    cmd = [sys.executable, str(shared_runner), *sys.argv[1:], "--tenants", "1"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
