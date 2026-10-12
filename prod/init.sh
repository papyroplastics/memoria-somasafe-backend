#!/bin/sh
set -e
dir=$(cd "$(dirname "$0")" && pwd)
backend=$(cd "$dir/.." && pwd)
registry=$(terraform -chdir="$dir/terraform" output -raw registry)
project=$(terraform -chdir="$dir/terraform" output -raw project)

cd "$backend"
make shared
podman build -f prod/api.Containerfile -t "$registry/somasafe-api:latest" .
podman build -f prod/worker.Containerfile --build-context calibration=shared/gen/calibration \
  -t "$registry/somasafe-worker:latest" .
podman build -f benchmark/bench.Containerfile -t "$registry/somasafe-bench:latest" .
gcloud auth print-access-token | podman login -u oauth2accesstoken --password-stdin "${registry%%/*}"
for image in api worker bench; do
  podman push "$registry/somasafe-$image:latest"
done
gcloud secrets versions add server-private-key --project "$project" \
  --data-file="$backend/shared/gen/server-private-key.pem"

until "$dir/seed.sh" "${1:-200}"; do
  echo "seeding failed, retrying in 15s (is connect.sh running?)"
  sleep 15
done
