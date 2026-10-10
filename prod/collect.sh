#!/bin/sh
set -e
run=${1:?usage: $0 <run_id>}
dir=$(cd "$(dirname "$0")" && pwd)
backend=$(cd "$dir/.." && pwd)
user=$(gcloud compute os-login describe-profile --format='value(posixAccounts[0].username)')
hosts=$(terraform -chdir="$dir/terraform" output -raw hosts)
ssh="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
remote=/opt/somasafe/backend/results
results="$backend/results"

pull() {
  address=$(echo "$hosts" | awk -v host="$1" '$1 == host && $2 == "RUNNING" { print $3 }')
  if [ -z "$address" ]; then
    echo "skipped $1, not running"
    return
  fi
  [ -z "$4" ] || rm -rf "$3"
  mkdir -p "$3"
  $ssh "$user@$address" sudo tar -C "$2" -cf - . | tar -C "$3" -xf -
}

pull client-1-inst "$remote/benchmark/$run" "$results/benchmark/$run"
make -C "$backend" prod-collect RUN="$run"

pull celery-1-inst "$remote/worker-metrics" "$results/worker-metrics"

snapshot=$(curl -sf -X POST localhost:9090/api/v1/admin/tsdb/snapshot \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"]["name"])')
pull client-1-inst "$remote/prometheus/snapshots/$snapshot" "$results/benchmark/tsdb" fresh
client=$(echo "$hosts" | awk '$1 == "client-1-inst" { print $3 }')
$ssh "$user@$client" sudo rm -rf "$remote/prometheus/snapshots/$snapshot"
