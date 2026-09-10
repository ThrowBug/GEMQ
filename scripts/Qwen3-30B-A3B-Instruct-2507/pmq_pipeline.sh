#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${script_dir}/pmq_compute_stats.sh"
bash "${script_dir}/pmq_allocate.sh"
bash "${script_dir}/pmq_quantize.sh"

