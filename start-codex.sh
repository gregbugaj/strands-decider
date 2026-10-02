#!/usr/bin/env bash
set -euo pipefail

# Keep the local launcher in full-access mode while preserving any Codex
# subcommands and arguments, such as `resume --last` or a session id.
exec codex --sandbox danger-full-access "$@"
