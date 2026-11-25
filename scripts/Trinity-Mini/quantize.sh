#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../model_extension.sh" "arcee-ai/Trinity-Mini" "quantize" "$@"
