#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../model_extension.sh" "zai-org/GLM-4.7-Flash" "quantize" "$@"
