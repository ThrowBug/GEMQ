#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../model_extension.sh" "mistralai/Mixtral-8x7B-v0.1" "compute_stats" "$@"
