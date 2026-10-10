#!/bin/sh
set -e
dir=$(cd "$(dirname "$0")" && pwd)
user=$(gcloud compute os-login describe-profile --format='value(posixAccounts[0].username)')
bastion=$(terraform -chdir="$dir/terraform" output -raw hosts | awk '$1 == "client-1-inst" && $2 == "RUNNING" { print $3 }')

echo "forwarding postgres, redis and prometheus through client-1-inst, ctrl-c to stop"
exec ssh -N -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 5432:postgres-inst:5432 -L 6379:redis-auth-inst:6379 \
  -L 9090:localhost:9090 "$user@$bastion"
