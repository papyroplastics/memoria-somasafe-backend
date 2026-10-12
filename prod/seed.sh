#!/bin/sh
set -e
dir=$(cd "$(dirname "$0")" && pwd)
backend=$(cd "$dir/.." && pwd)
. "$dir/host-env.sh"
make -C "$backend" shared
cd "$backend"
exec uv run -m scripts.system.seed_db --assign-device --test-users "${1:-200}"
