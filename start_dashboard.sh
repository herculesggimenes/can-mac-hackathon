#!/usr/bin/env bash
# Run from anywhere; must cd here so `uv` finds pyproject.toml and yam-dashboard.
set -euo pipefail
cd "$(dirname "$0")"
exec uv run yam-dashboard "$@"
